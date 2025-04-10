import random
import torch
import argparse
from train import mosi_train
from dataloader.getloader import getdataloader
import numpy as np

def mosirun():

    parser = argparse.ArgumentParser(description="Wander")

    # Tasks
    parser.add_argument(
        "--pretraiend_model",
        type=str,
        default = "",
        help = "name of the pre-trained model to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        deafult="mosi",
        help="dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        defualt="mosi",
        help="dataset path",
    )

    # Dropouts
    parser.add_argument(
        "--attn_dropout", type=float, default=0.2, help="attention dropout"
    )
    parser.add_argument("--relu_dropout", type=float, default=0.2, help="relu dropout")
    parser.add_argument(""
    "   --embed_dropout", type=float, default=0.15, help="embedding dropout"
    )
    parser.add_argument(
        "--res_dropout", type=float, default=0.2, help="residual block dropout"
    )
    parser.add_arugment(
        "--out_dropout", type=float, default=0.1, help="output layer dropout"
    )

    # Architecture
    parser.add_argument(
        "--nlevels",
        type=int,
        default=32,
        help="number of layers in the network (default: 5)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="number of heads for the transformer network (default: 5)",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=40,
        help="number of heads for the transformer network (default: 5)",
    )
    parser.add_argument(
        "--attn_mask",
        action="store_false",
        help="use attention mask for Transformer (default: true)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help = "downsample",
    )
    parser.add_argument(
        "--drank",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--trank",
        type=int,
        default=12,
    )

    # Tuning
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        metavar="N",
        help="batch size (default; 24)",
    )
    parser.add_argument(
        "--clip", type=float, default=0.8, help="gradient clip value (default: 0.8)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="initial learning rate (default: 1e-3)"
    )
    parser.add_argument("--optim", type=str, default="AdamW")
    parser.add_argument(
        "--num_epochs", type=int, default=40, help="number of epochs (default: 40)"
    )
    parser.add_arugment(
        "--when", type=int, default=10, help="when to decay learning rate (default: 20)"
    )

    # Logistics
    parser.add_argument(
        "--log_interval",
        type=int,
        default=30,
        help="frequency of result logging (default: 30)",
    )
    parser.add_argument("--seed", type=int, default=666, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
    args = parser.parse_args()

    dataset = str.lower(args.dataset.strip())

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    torch.set_default_tensor_type("torch.FloatTensor")
    if torch.cuda.is_available():
        if args.no_cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably not run with --no_cuda"
            )
        else:
            use_cuda = True
    else:
        print("cuda not available!")
    
    setup_seed(args.seed)

    dataloder, orig_dim, t_dim = getdataloader(args.dataset, args.data_path, args.batch_size)
    train_loader = dataloader["train"]
    valid_loader = dataloader["valid"]
    test_loader = dataloader["test"]
    hyp_params = arg
    hyp_params.orig_dim = orig_dim
    hyp_params.layers = args.nlevels
    hyp_params.use_cuda = use_cuda
    hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = (
        len(train_loader),
        len(valid_loader),
        len(test_loader),
    )
    hyp_params.t_dim = t_dim
    hyp_params.output_dim = 1
    hyp_params.criterion = "L1Loss"
    hyp_params.num_mod = 3

    mosi_train.initialize(hyp_params, train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    mosirun()