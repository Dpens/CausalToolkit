from torch.utils.data import Dataset

class timeseries(Dataset):
    def __init__(self, x, y, device):
        self.x = x.to(device)
        self.y = y.to(device)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len
