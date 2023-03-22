import torch
from torch.utils.data import Dataset

import gdown

class ImgDataset(Dataset):
    def __init__(self, embeddings, transform=None):
        self.embeddings = embeddings
        self.transform = transform
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.embeddings[idx])
        else:
            return self.embeddings[idx]
        
        
def download_from_drive(url:str, file_path:str):
    gdown.download(url, file_path, quiet=False)