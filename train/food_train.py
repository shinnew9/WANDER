import torch
from torch import nn
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.foodmodel import FoodModelWander
from utils.eval_metrics import eval_food
from utils.util import transfer_model, get_parameter_number

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = FoodModelWander(
        hyp_params.pretrained_vit,
        hyp_params.pretrained_text,
        hyp_params.output_dim,
        hyp_params.t_dim,
        hyp_params.rank,
        hyp_params.drank,
        hyp_params.trank,
        hyp_params.out_dropout
    )

    transfer_model(hyp_params.pretrained_model, model)
    print(get_parameter_number(model))

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(
        model.parameter(), lr = hyp_params.lr, weight_decay = 4e-5
    )

    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience = hyp_params.when, factor=0.5, verbose=True
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
        model.train()
        for i_batch, batch in enumerate(train_loader):
            text, image, batch_Y = batch
            eval_attr = batch_Y.squeeze(-1)  # if num of labels is 1
            model.zero_grad()
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    tia, ta, tt = (
                        text["input_ids"].cuda(),
                        text["attention_mask"].cuda(),
                        text["token_type_ids"].cuda(),
                    )
                    image, eval_attr = image.cuda(), eval_attr.cuda()
                    eval_attr = eval_attr.long()
            
            batch_size = image.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model

            preds = net(image, [tia, ta, tt])
            preds = preds.sview(-1, 101)
            eval_attr = eval_attr.view(-1)

            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)

            optimizer.step()

        
        def evaluate(model, criterion, test=False):
            model.eval()
            loader = test_loader if test else valid_loader
            total_loss = 0.0

            results = []
            truths = []

            with torch.no_grad():
                for i_batch, batch in enumerate(loader):
                    text, image, batch_Y = batch
                    eval_attr = batch_Y.squeeze(dim=-1)   # if num of labels is 1

                    if hyp_params.use_cuda:
                        with torch.cuda.device(0):
                            ti, ta, tt = (
                                text["input_ids"].cuda(),
                                text["attention_mask"].cuda(),
                                text["token_tpye_ids"].cuda(), 
                            )
                            image, eval_attr = image.cuda(), eval_attr.cuda()
                            eval_attr = eval_attr.long()

                    net = model
                    preds = net(image, [tia, ta, tt])
                    preds = preds.view(-1, 101)
                    eval_attr = eval_attr.view(-1)
                    total_loss = criterion(preds, eval_attr).item()

                    # Collect the results into dictionary
                    results.append(preds)
                    truths.append(eval_attr)

            avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths
        
        best_acc = 0
        for epoch in range(1, hyp_params.num_epochs + 1):
            start = time.time()
            train(model, optimizer, criterion)
            val_loss, r, t = evaluate(model, criterion, test=False)
            acc = eval_food(r, t)

            end = time.time()
            duration = end - start
            scheduler.step(val_loss)  # Decay learning rate by validation loss
            
            print("-" * 50)
            print(
                "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f}".format(
                    epoch, duration, val_loss
                )
            )
            print("-" * 50)

            if acc > best_acc:
                best_acc = acc

        print("Best accuracy of validation: {:.4f}".format(best_acc))