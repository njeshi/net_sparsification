import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from dataset import SSTDataset
from models import BertForSentimentClassification
import matplotlib.pyplot as plt
import logging

# create logger
logger = logging.getLogger('no_spam')
logger.setLevel(logging.DEBUG)

def load_sst_dataset(dataset_path, batch_size, num_workers, 
                     maxlen_train=256, maxlen_val=256, 
                     model_path=None, return_sentences=False):

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        train_set = SSTDataset(filename=os.path.join(dataset_path, 'train.tsv'), 
                               maxlen=maxlen_train, 
                               tokenizer=tokenizer,
                               return_sentences = return_sentences)
        
        test_set = SSTDataset(filename=os.path.join(dataset_path, 'test.tsv'), 
                              maxlen=maxlen_val, 
                              tokenizer=tokenizer,
                              return_sentences = return_sentences)
        
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size, 
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=test_set, 
                                 batch_size=batch_size, 
                                 num_workers=num_workers)
        #assert len(np.unique(train_set.df['label'].values)) == len(np.unique(test_set.df['label'].values))
        train_set.num_classes = 2
        # train_loader.dataset.targets = train_loader.dataset.df['label'].values
        # test_loader.dataset.targets = test_loader.dataset.df['label'].values
        
        return train_set, train_loader, test_loader

def load_bert_model(model_path, device='cuda'):
    """Loads bert for sentiment classification model.
    Args:
        model_path (str): Path to model
        device (str): Device on which to keep the model
    Returns:
        model: Torch model
    """
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type == 'bert':
        model = BertForSentimentClassification.from_pretrained(model_path)
    else:
        raise ValueError('This transformer model is not supported yet.')
        
    model.eval()
    model = nn.DataParallel(model.to(device))
    return model




def load_glm(result_dir):
    
    Nlambda = max([int(f.split('params')[1].split('.pth')[0]) 
                   for f in os.listdir(result_dir) if 'params' in f]) + 1
    
    print(f"Loading regularization path of length {Nlambda}")
    
    params_dict = {i: torch.load(os.path.join(result_dir, f"params{i}.pth"),
                          map_location=torch.device('cpu')) for i in range(Nlambda)}
    
    regularization_strengths = [params_dict[i]['lam'].item() for i in range(Nlambda)]
    weights = [params_dict[i]['weight'] for i in range(Nlambda)]
    biases = [params_dict[i]['bias'] for i in range(Nlambda)]
   
    metrics = {'acc_tr': [], 'acc_val': [], 'acc_test': []}
    
    for k in metrics.keys():
        for i in range(Nlambda):
            metrics[k].append(params_dict[i]['metrics'][k])
        metrics[k] = 100 * np.stack(metrics[k])
    metrics = pd.DataFrame(metrics)
    metrics = metrics.rename(columns={'acc_tr': 'acc_train'})
    
    weights_stacked = torch.stack(weights)
    sparsity = torch.sum(weights_stacked != 0, dim=2).numpy()
            
    return {'metrics': metrics, 
            'regularization_strengths': regularization_strengths, 
            'weights': weights, 
            'biases': biases,
            'sparsity': sparsity,
            'weight_dense': weights[-1],
            'bias_dense': biases[-1]}

def select_sparse_model(result_dict, 
                        selection_criterion='absolute', 
                        factor=6):
    
    assert selection_criterion in ['sparsity', 'absolute', 'relative', 'percentile'] 

    metrics, sparsity = result_dict['metrics'], result_dict['sparsity']
    
    acc_val, acc_test = metrics['acc_val'], metrics['acc_test']

    if factor == 0:
        sel_idx = -1
    elif selection_criterion == 'sparsity':
        sel_idx = np.argmin(np.abs(np.mean(sparsity, axis=1) - factor))
    elif selection_criterion == 'relative':
        sel_idx = np.argmin(np.abs(acc_val - factor * np.max(acc_val)))
    elif selection_criterion == 'absolute':
        delta = acc_val - (np.max(acc_val) - factor)
        lidx = np.where(delta <= 0)[0]
        sel_idx = lidx[np.argmin(-delta[lidx])]
    elif selection_criterion == 'percentile': 
        diff = np.max(acc_val) - np.min(acc_val)
        sel_idx = np.argmax(acc_val >  np.max(acc_val) - factor * diff)

    print(f"Test accuracy | Best: {max(acc_test): .2f},",
          f"Sparse: {acc_test[sel_idx]:.2f}",
          f"Sparsity: {np.mean(sparsity[sel_idx]):.2f}")

    result_dict.update({'weight_sparse': result_dict['weights'][sel_idx], 
                        'bias_sparse': result_dict['biases'][sel_idx]})
    return result_dict




def plot_sparsity(results):
    """Function to visualize the sparsity-accuracy trade-off of regularized decision
    layers
    Args:
        results (dictionary): Appropriately formatted dictionary with regularization
        paths and logs of train/val/test accuracy.
    """
        
    if type(results['metrics']['acc_train'].values[0]) == list:
        all_tr = 100 * np.array(results['metrics']['acc_train'].values[0])
        all_val = 100 * np.array(results['metrics']['acc_val'].values[0])
        all_te = 100 * np.array(results['metrics']['acc_test'].values[0])
    else:
        all_tr = 100 * np.array(results['metrics']['acc_train'].values)
        all_val = 100 * np.array(results['metrics']['acc_val'].values)
        all_te = 100 * np.array(results['metrics']['acc_test'].values)

    fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
    axarr[0].plot(all_tr)
    axarr[0].plot(all_val)
    axarr[0].plot(all_te)
    axarr[0].legend(['Train', 'Val', 'Test'], fontsize=16)
    axarr[0].set_ylabel("Accuracy (%)", fontsize=18)
    axarr[0].set_xlabel("Regularization index", fontsize=18)

    num_features = results['weights'][0].shape[1]
    total_sparsity = np.mean(results['sparsity'], axis=1) / num_features
    axarr[1].plot(total_sparsity, all_tr, 'o-')
    axarr[1].plot(total_sparsity, all_te, 'o-')
    axarr[1].legend(['Train', 'Val', 'Test'], fontsize=16)
    axarr[1].set_ylabel("Accuracy (%)", fontsize=18)
    axarr[1].set_xlabel("1 - Sparsity", fontsize=18)
    axarr[1].set_xscale('log')
    
    plt.show()

   
    


