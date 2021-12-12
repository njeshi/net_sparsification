import os
import math
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset, random_split
from dataset import add_index_to_dataloader

def _balance_dataset(dataset): 
    """Balances a given dataset to have the same number of samples/class.
    Args:
        dataset : Torch dataset
    Returns:
        Torch dataset with equal number of samples/class
    """
    
    print("Balancing dataset...")
    n = len(dataset)
    labels = torch.Tensor([dataset[i][1] for i in range(n)]).int()
    n0 = sum(labels).item()
    I_pos = labels == 1

    idx = torch.arange(n)
    idx_pos = idx[I_pos]
    torch.manual_seed(0)
    I = torch.randperm(n - n0)[:n0]
    idx_neg = idx[~I_pos][I]
    idx_bal = torch.cat([idx_pos, idx_neg],dim=0)
    return Subset(dataset, idx_bal)

def _load_features(feature_path):
    """Loads precomputed deep features.
    Args:
        feature_path (str): Path to precomputed deep features
        
    Returns:
        Torch dataset with recovered deep features.
    """
    if not os.path.exists(os.path.join(feature_path, f"0_features.npy")): 
        raise ValueError(f"The provided location {feature_path} does not contain any representation files")

    ds_list, chunk_id = [], 0
    while os.path.exists(os.path.join(feature_path, f"{chunk_id}_features.npy")): 
        features = torch.from_numpy(np.load(os.path.join(feature_path, f"{chunk_id}_features.npy"))).float()
        labels = torch.from_numpy(np.load(os.path.join(feature_path, f"{chunk_id}_labels.npy"))).long()
        ds_list.append(TensorDataset(features, labels))
        chunk_id += 1

    print(f"==> loaded {chunk_id} files of representations...")
    return ConcatDataset(ds_list)

def _compute_features(loader, model, batch_size, num_workers, 
                     shuffle=False, device='cuda', filename=None, 
                     chunk_threshold=20000, balance=False):
    
    """Compute deep features for a given dataset using a modeln and returnss
    them as a pytorch dataset and loader. 
    Args:
        loader : Torch data loader
        model: Torch model
        dataset_type (str): One of vision or language
        pooled_output (bool): Whether or not to pool outputs
          (only relevant for some language models)
        batch_size (int): Batch size for output loader
        num_workers (int): Number of workers to use for output loader
        shuffle (bool): Whether or not to shuffle output data loaoder
        device (str): Device on which to keep the model
        filename (str):Optional file to cache computed feature. Recommended
            for large datasets like ImageNet.
        chunk_threshold (int): Size of shard while caching
        balance (bool): Whether or not to balance output data loader 
            (only relevant for some language models)
    Returns:
        feature_dataset: Torch dataset with deep features
        feature_loader: Torch data loader with deep features
    """
    
    if filename is None or not os.path.exists(os.path.join(filename, f'0_features.npy')):

        all_latents, all_targets = [], []
        Nsamples, chunk_id = 0, 0 
        
        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)): 
            
            with torch.no_grad():
                (input_ids, attention_mask, targets) = batch
                mask = targets != -1
                input_ids, attention_mask, targets = [\
                    t[mask] for t in (input_ids, attention_mask, targets)]
                
                if hasattr(model, 'module'):
                    model = model.module
                latents = model.bert(input_ids=input_ids.to(device), 
                                    attention_mask=attention_mask.to(device))

                latents = latents[0][:,0]
            
            if batch_idx == 0:
                print("Latents shape", latents.shape)
                
                
            Nsamples += latents.size(0)

            all_latents.append(latents.cpu())
            all_targets.append(targets.cpu())

            if filename is not None and Nsamples > chunk_threshold: 
                if not os.path.exists(filename): os.makedirs(filename)
                np.save(os.path.join(filename, f'{chunk_id}_features.npy'), torch.cat(all_latents).numpy())
                np.save(os.path.join(filename, f'{chunk_id}_labels.npy'), torch.cat(all_targets).numpy())
                all_latents, all_targets, Nsamples = [], [], 0
                chunk_id += 1
                
        if filename is not None and Nsamples > 0:
            if not os.path.exists(filename): os.makedirs(filename)
            np.save(os.path.join(filename, f'{chunk_id}_features.npy'), torch.cat(all_latents).numpy())
            np.save(os.path.join(filename, f'{chunk_id}_labels.npy'), torch.cat(all_targets).numpy())


    feature_dataset = _load_features(filename) if filename is not None else \
        TensorDataset(torch.cat(all_latents), torch.cat(all_targets))
    if balance: 
        feature_dataset = _balance_dataset(feature_dataset)
        
    feature_loader = DataLoader(feature_dataset, 
                                       num_workers=num_workers,
                                       batch_size=batch_size, 
                                       shuffle=shuffle)
    
    return feature_dataset, feature_loader

def calculate_metadata(loader, num_classes=None, filename=None): 
    """Calculates mean and standard deviation of the deep features over
    a given set of images.
    Args:
        loader : torch data loader
        num_classes (int): Number of classes in the dataset
        filename (str): Optional filepath to cache metadata. Recommended
            for large datasets like ImageNet.
        
    Returns:
        metadata (dict): Dictionary with desired statistics.
    """
    
    if filename is not None and os.path.exists(filename):
        return torch.load(filename)
    
    # Calculate number of classes if not given
    if num_classes is None: 
        num_classes = 1
        for batch in loader:
            y = batch[1]
            print(y)
            num_classes = max(num_classes, y.max().item()+1)
            
    eye = torch.eye(num_classes)

    X_bar, y_bar, y_max, n = 0, 0, 0, 0
 
    # calculate means and maximum
    print("Calculating means")
    for X,y in tqdm(loader, total=len(loader)):
        X_bar += X.sum(0)
        y_bar += eye[y].sum(0)
        y_max = max(y_max, y.max())
        n += y.size(0)
    X_bar = X_bar.float()/n
    y_bar = y_bar.float()/n

    # calculate std
    X_std, y_std = 0, 0
    print("Calculating standard deviations")
    for X,y in tqdm(loader, total=len(loader)): 
        X_std += ((X - X_bar)**2).sum(0)
        y_std += ((eye[y] - y_bar)**2).sum(0)
    X_std = torch.sqrt(X_std.float()/n)
    y_std = torch.sqrt(y_std.float()/n)

    # calculate maximum regularization
    inner_products = 0
    print("Calculating maximum lambda")
    for X,y in tqdm(loader, total=len(loader)): 
        y_map = (eye[y] - y_bar)/y_std
        inner_products += X.t().mm(y_map)*y_std

    inner_products_group = inner_products.norm(p=2,dim=1)

    metadata = {
                "X": {
                        "mean": X_bar, 
                        "std": X_std, 
                        "num_features": X.size()[1:], 
                        "num_examples": n
                    }, 
                "y": {
                        "mean": y_bar, 
                        "std": y_std,
                        "num_classes": y_max+1
                    },
                "max_reg": {
                        "group": inner_products_group.abs().max().item()/n, 
                        "nongrouped": inner_products.abs().max().item()/n
                    }
                }
    
    if filename is not None:
        torch.save(metadata, filename)
    
    return metadata

def _split_dataset(dataset, Ntotal, val_frac,
                  batch_size, num_workers,
                 random_seed=0, shuffle=True, balance=False):
    """Splits a given dataset into train and validation
    Args:
        dataset : Torch dataset
        Ntotal: Total number of dataset samples
        val_frac: Fraction to reserve for validation
        batch_size (int): Batch size for output loader
        num_workers (int): Number of workers to use for output loader
        random_seed (int): Random seed
        shuffle (bool): Whether or not to shuffle output data loaoder
        balance (bool): Whether or not to balance output data loader 
            (only relevant for some language models)
        
    Returns:
        split_datasets (list): List of datasets (one each for train and val)
        split_loaders (list): List of loaders (one each for train and val)
    """
    
    Nval = math.floor(Ntotal*val_frac)
    train_ds, val_ds = random_split(dataset, 
                                    [Ntotal - Nval, Nval], 
                                    generator=torch.Generator().manual_seed(random_seed))
    if balance: 
        val_ds = _balance_dataset(val_ds)
    split_datasets = [train_ds, val_ds]
        
    split_loaders = []
    for ds in split_datasets:
        split_loaders.append(DataLoader(ds, 
                                        num_workers=num_workers, 
                                        batch_size=batch_size, 
                                        shuffle=shuffle))
    return split_datasets, split_loaders



def compute_features(model, train_loader, test_loader, num_classes, out_dir_feats, \
                     batch_size=256, balance=True, device='cuda', num_workers=os.cpu_count()):
    print('--------- Computing/loading deep features ---------')
    feature_loaders = {}
    Ntotal = len(train_loader.dataset)

    for mode, loader in zip(['train', 'test'], [train_loader, test_loader]): 
        print(f"For {mode} set...")

        sink_path = os.path.join(out_dir_feats, f'features_{mode}')
        metadata_path = os.path.join(out_dir_feats, f'metadata_{mode}.pth')

        feature_ds, feature_loader = _compute_features(loader, model, batch_size=batch_size, 
                                                       shuffle=(mode == 'test'),
                                                       device=device,
                                                       filename=sink_path, 
                                                       num_workers=num_workers,
                                                       balance=balance if mode == 'test' else False)
        
        if mode == 'train':
            metadata = calculate_metadata(feature_loader, 
                                          num_classes=num_classes, 
                                          filename=metadata_path)

            split_datasets, split_loaders = _split_dataset(feature_ds, 
                                                           Ntotal, 
                                                           val_frac=0.1,
                                                           batch_size=batch_size, 
                                                           num_workers=num_workers,
                                                           random_seed=0, 
                                                           shuffle=True, 
                                                           balance=balance)

            feature_loaders.update(
                {mm : add_index_to_dataloader(split_loaders[mi])
                for mi, mm in enumerate(['train', 'val'])})
            
        else:
            feature_loaders[mode] = feature_loader

        
    return feature_loaders, metadata



def load_features(feature_path):
    """Loads precomputed deep features.
    Args:
        feature_path (str): Path to precomputed deep features
        
    Returns:
        Torch dataset with recovered deep features.
    """
    if not os.path.exists(os.path.join(feature_path, f"0_features.npy")): 
        raise ValueError(f"The provided location {feature_path} does not contain any representation files")

    ds_list, chunk_id = [], 0
    while os.path.exists(os.path.join(feature_path, f"{chunk_id}_features.npy")): 
        features = torch.from_numpy(np.load(os.path.join(feature_path, f"{chunk_id}_features.npy"))).float()
        labels = torch.from_numpy(np.load(os.path.join(feature_path, f"{chunk_id}_labels.npy"))).long()
        ds_list.append(TensorDataset(features, labels))
        chunk_id += 1

    print(f"==> loaded {chunk_id} files of representations...")
    return ConcatDataset(ds_list)
