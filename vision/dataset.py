import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from robustness.datasets import CIFAR    
    

class IndexedTensorDataset(TensorDataset): 
    def __getitem__(self, index): 
        val = super(IndexedTensorDataset, self).__getitem__(index)
        return val + (index,)
    
class IndexedDataset(Dataset): 
    def __init__(self, ds, sample_weight=None): 
        super(Dataset, self).__init__()
        self.dataset = ds
        self.sample_weight=sample_weight
    
    def __getitem__(self, index): 
        val = self.dataset[index]
        if self.sample_weight is None: 
            return val + (index,)
        else: 
            weight = self.sample_weight[index]
            return val + (weight,index)
    def __len__(self): 
        return len(self.dataset)


class NormalizedRepresentation(torch.nn.Module): 
    def __init__(self, loader, metadata, device='cuda', tol=1e-5): 
        super(NormalizedRepresentation, self).__init__()

        assert metadata is not None
        self.device = device
        self.mu = metadata['X']['mean']
        self.sigma = torch.clamp(metadata['X']['std'], tol)

    def forward(self, X): 
        return (X - self.mu.to(self.device))/self.sigma.to(self.device)
    

def add_index_to_dataloader(loader, sample_weight=None): 
    return DataLoader(IndexedDataset(loader.dataset, sample_weight=sample_weight), 
                      batch_size=loader.batch_size, 
                      sampler=loader.sampler, 
                      num_workers=loader.num_workers, 
                      collate_fn=loader.collate_fn, 
                      pin_memory=loader.pin_memory, 
                      drop_last=loader.drop_last, 
                      timeout=loader.timeout, 
                      worker_init_fn=loader.worker_init_fn, 
                      multiprocessing_context=loader.multiprocessing_context
                      )