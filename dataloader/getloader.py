from torch.utils.data import DataLoader

from dataloader.mosidata import CMUData
from dataloader.fooddata import Food101


def getdataloader(dataset, data_path, batch_size):
    if dataset == "mosi":
        data = {
            "train": CMUData(data_path, "train"),
            "valid": CMUData(data_path, "valid"),
            "test": CMUData(data_path, "test"), 
        }
        orig_dim = data["test"].get_dim()
        t_dim = data["test"].get_tim()  # train_dimension이라 t_dim인가?
        dataLoader = {
            ds: DataLoader(data[ds], batch_size = batch_size, num_workers = 8)
            for ds in data.keys() 
        }
    elif dataset == "food":
        data = {
            "train": Food101(mode="train", dataset_root_dir = data_path),
            "valid": Food101(mode="valid", dataset_root_dir = data_path),
            "test": Food101(mode="test", dataset_root_dir = data_path), 
        }
        orig_dim, t_dim = None, None
        dataLoader = {
            ds: DataLoader(data[ds], batch_size = batch_size, num_workers=8)
            for ds in data.keys()
        }

    return dataLoader, orig_dim, t_dim