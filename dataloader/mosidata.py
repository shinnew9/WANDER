from torch.utils.data.dataset import Dataset
import pickle
import torch

class CMUData(Dataset):
    def __init__(self, data_path, split):
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        self.data = data[split]

        self.split = split
        self.orig_dims = [
            self.data['text'][0].shape[1],
            self.data['audio'][0].shape[1],
            self.data['vision'][0].shape[1]
        ]

    def get_dim(self):
        return self.orig_dims
    
    def get_tim(self):
        return [self.data['text'][0].shape[0],
                self.data['audio'][0].shape[0],
                self.data['vision'][0].shape[0]]

    def __len__(self):
        return self.data['audio'].shape[0]
    
    def __getitem__(self, idx):
        return {
            'audio': torch.tensor(self.data['audio'][idx]).float(),
            'vision': torch.tensor(self.data['vision'][idx]).float(),
            'text': torch.tensor(self.data['text'][idx]).float(),
            'labels': torch.tensor(self.data['regression_labels'][idx]).float(), 
        }