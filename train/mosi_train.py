import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.model import AdapterModel
from utils.eval_metrics import eval_senti
from utils.util import transfer_model, get_parameter_number
import time


def initialize(hyp_params, train_loader, valid_loader, test_loader):
    model = AdapterModel(
        hyp_params.orig_dim,
        hyp_params.t_dim,
        hyp_params.rank,
        hyp_params.drank,
        hyp_params.trank,
        hyp_params.output_dim,
        hyp_params.proj_dim,
        hyp_params.num_heads,
        hyp_params.layers,
        hyp_params.relu_dropout,
        hyp_params.embed_dropout,
        hyp_params.res_dropout,
        hyp_params.out_dropout,
        hyp_params.attn_dropout,
    )
    transfer_model(hyp_params.pretrained_model, model)
    print(get_parameter_number(model))

    if hyp_params.use_cuda:
        model = model.cuda()
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1, verbose=True
    )
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        proc_loss, proc_size = 0, 0
        for i_batch, batch in enumerate(train_loader):
            text, audio, vision, batch_Y = (
                batch["text"],
                batch["audio"],
                batch["vision"],
                batch["labels"], 
            )

            eval_attr = batch_Y.unsqueeze(-1)
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = (
                        text.cuda(),
                        audio.cuda(),
                        vision.cuda(),
                        eval_attr.cuda(), 
                    )

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds = net([text, audio, vision])

            raw_loss = criterion(preds, eval_attr)
            combined_loss = raw_loss
            combined_loss.backward()

            torch.nn.utils.clip_gard_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item()*batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item()*batch_size

        return epoch_loss / hyp_params.n_train
    
    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
                text, audio, vision, batch_Y = (
                    batch["text"],
                    batch["audio"],
                    batch["vision"],
                    batch["labels"],
                )
                eval_attr = batch_Y.unsqueeze(-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = (
                            text.cuda(),
                            audio.cuda(),
                            vision.cuda(),
                            eval_attr.cuda(),
                        )
                batch_size = text.size(0)

                net = nn.DataParallel(model) if batch_size > 10 else model
                preds = net([text, audio, vision])
                total_loss += criterion(preds, eval_attr).item()

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
    
    best_acc = 0
    for epoch in range(1, hyp_params.num_epochs +1):
        start = time.time()
        train(model, optimizer, criterion)
        val_loss, r, t = evaluate(model, criterion, test=False)
        acc2 = eval_senti(r, t)

        end = time.time()
        duration = end - start 
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {5.4f} sec | Valid Loss {:5.4f}".format(
                epoch, duration, val_loss  
            )
        )

        print("-" * 50)

        if best_acc < acc2:
            best_acc = acc2
    print("Best Accruacy of validation: ", best_acc)
