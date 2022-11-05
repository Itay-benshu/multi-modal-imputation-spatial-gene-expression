import torch
import numpy as np
from skimage import io
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
toTensor = transforms.ToTensor()
resize = transforms.Resize(224)

class SpotAutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_spots, n_genes):
        super().__init__()
        
        # R
        self.R = np.zeros((n_spots, n_genes))
        self.R[data[:, 1], data[:, 0]] = data[:, 2]
        
        # Mask
        self.mask = (self.R > 0).astype(float)

    def __getitem__(self, index):
        return (self.R[index], self.mask[index], index)

    def __len__(self):
        return self.R.shape[0]

class SpotPositionDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_spots, n_genes, spot_positions):
        super().__init__()
        
        # R
        self.y = np.zeros((n_spots, n_genes))
        self.y[data[:, 1], data[:, 0]] = data[:, 2]
        
        self.x = spot_positions
        self.mask = (self.y > 0).astype(float)

    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.mask[index])

    def __len__(self):
        return self.x.shape[0]


class SpotImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_spots, n_genes, tile_paths):
        super().__init__()
        
        # R
        self.y = np.zeros((n_spots, n_genes))
        self.y[data[:, 1], data[:, 0]] = data[:, 2]
        
        self.x = tile_paths
        self.mask = (self.y > 0).astype(float)

    def __getitem__(self, index):
        return (resize(toTensor(io.imread(self.x[index]))), self.y[index], self.mask[index])

    def __len__(self):
        return len(self.x)
    
    
class FinalModelDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_spots, n_genes, spot_positions, tile_paths):
        super().__init__()
        
        # R
        self.R = np.zeros((n_spots, n_genes))
        self.R[data[:, 1], data[:, 0]] = data[:, 2]
        
        # Position
        self.position_info = spot_positions
        
        # Image
        self.tile_paths = tile_paths
        
        # Mask
        self.mask = (self.R > 0).astype(float)

    def __getitem__(self, index):
        return (self.R[index], 
                self.mask[index], 
                self.position_info[index],
                resize(toTensor(io.imread(self.tile_paths[index]))), 
                index)

    def __len__(self):
        return self.R.shape[0]