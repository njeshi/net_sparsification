import os
import torch
import torchvision
import numpy as np
import pandas as pd
from robustness.datasets import DATASETS as VISION_DATASETS
from robustness.model_utils import make_and_restore_model
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING) 

from lime import lime_image
from functools import partial



def load_dataset(dataset_name, dataset_path, batch_size, num_workers,
                 maxlen_train=256, maxlen_val=256, shuffle=False, 
                 model_path=None, return_sentences=False):

        if dataset_name == 'places-10': dataset_name = 'places365'        
        if dataset_name not in VISION_DATASETS:
            raise ValueError("Vision dataset not currently supported...")
        dataset = VISION_DATASETS[dataset_name](os.path.expandvars(dataset_path))
        
        if dataset_name == 'places365': 
            dataset.num_classes = 10
        
        train_loader, test_loader = dataset.make_loaders(num_workers, 
                                                        batch_size, 
                                                        data_aug=False, 
                                                        shuffle_train=shuffle, 
                                                        shuffle_val=shuffle)
        return dataset, train_loader, test_loader
    
    
def load_model(arch, dataset, model_root=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = make_and_restore_model(arch=arch, 
                                      dataset=dataset,
                                      resume_path=model_root, 
                                      pytorch_pretrained=(model_root is None))
    model.eval()
    model = torch.nn.DataParallel(model.to(device))
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
        results (dictionary): 
            Appropriately formatted dictionary with regularization 
            paths and logs of train/val/test accuracy.
        res_mult: int
            result multiplier, multiples
    """
        
    if type(results['metrics']['acc_train'].values[0]) == list:
        all_tr  = np.array(results['metrics']['acc_train'].values[0])
        all_val = np.array(results['metrics']['acc_val'].values[0])
        all_te  = np.array(results['metrics']['acc_test'].values[0])
    else:
        all_tr  = np.array(results['metrics']['acc_train'].values)
        all_val = np.array(results['metrics']['acc_val'].values)
        all_te  = np.array(results['metrics']['acc_test'].values)

    fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
    axarr[0].plot(all_tr, label='Train')
    axarr[0].plot(all_val, label='Val')
    axarr[0].plot(all_te, label='Test')
    axarr[0].legend(fontsize=16)
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
    
    
    


# ----------- LIME VISUALIZER HELPERS  ----------- #

    
def latent_predict(images, model, mean=None, std=None):
    """LIME helper function that computes the deep feature representation 
    for a given batch of images.
    Args:
        image (tensor): batch of images.
        model: deep network whose deep features are to be visualized.
        mean (tensor): mean of deep features.
        std (tensor): std deviation of deep features.
    Returns:
        Normalized deep features for batch of images.
    """
    preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])    
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0).to(device)
    
    (_, latents), _ = model(batch.to(torch.float), with_latent=True)
    scaled_latents = (latents.detach().cpu() - mean.to(torch.float)) / std.to(torch.float)
    return scaled_latents.numpy()
    
def parse_lime_explanation(expln, f, sign, NLime=3):
    """LIME helper function that extracts a mask from a lime explanation
    Args:
        expln: LIME explanation from LIME library
        f: indice of features to visualize
        sign: +/-1 array indicating whether the feature should be maximized/minimized.
        images (tensor): batch of images.
        NLime (int): Number of top-superpixels to visualize.
    Returns:
        Tensor where the first and second channels contains superpixels that cause the
        deep feature to activate and deactivate respectively.
    """
    segs = expln.segments
    vis_mask = np.zeros(segs.shape + (3,))

    weights = sorted([v for v in expln.local_exp[f]], 
                     key=lambda x: -np.abs(x[1]))
    weight_values = [w[1] for w in weights]
    pos_lim, neg_lim = np.max(weight_values), (1e-8 + np.min(weight_values))

    if NLime is not None:
        weights = weights[:NLime]

    for wi, w in enumerate(weights):
        if w[1] >= 0:
            si = (w[1] / pos_lim, 0, 0) if sign == 1 else (0, w[1] / pos_lim, 0)
        else:
            si = (0, w[1] / neg_lim, 0) if sign == 1 else (w[1] / neg_lim, 0, 0) 
        vis_mask[segs == w[0]] = si

    return torch.tensor(vis_mask.transpose(2, 0, 1))

def get_lime_explanation(model, feature_idx, signs,
                         images, rep_mean, rep_std,  
                         num_samples=1000,
                         NLime=3,
                         background_color=0.6):
    """Computes LIME explanations for a given set of deep features. The LIME
    objective in this case is to identify the superpixels within the specified
    images that maximally/minimally activate the corresponding deep feature.
    Args:
        model: deep network whose deep features are to be visualized.
        feature_idx: indice of features to visualize
        signs: +/-1 array indicating whether a feature should be maximized/minimized.
        images (tensor): batch of images.
        rep_mean (tensor): mean of deep features.
        rep_std (tensor): std deviation of deep features.
        NLime (int): Number of top-superpixels to visualize
        background_color (float): Color to assign non-relevant super pixels
    Returns:
        Tensor comprising LIME explanations for the given set of deep features.
    """
    explainer = lime_image.LimeImageExplainer(verbose=False)
    lime_objective = partial(latent_predict, model=model, mean=rep_mean, std=rep_std)

    explanations = []
    for im, feature, sign in zip(images, feature_idx, signs):
        explanation = explainer.explain_instance(im.numpy().transpose(1, 2, 0), 
                                     lime_objective, 
                                     labels=np.array([feature]), 
                                     top_labels=None,
                                     hide_color=0, 
                                     num_samples=num_samples) 
        explanation = parse_lime_explanation(explanation, 
                                             feature, 
                                             sign, 
                                             NLime=NLime)
        
        if sign == 1:
            explanation = explanation[:1].unsqueeze(0).repeat(1, 3, 1, 1)
        else:
            explanation = explanation[1:2].unsqueeze(0).repeat(1, 3, 1, 1)
        
        interpolated = im * explanation + background_color * torch.ones_like(im) * (1 - explanation)
        explanations.append(interpolated)
        
    return torch.cat(explanations)


def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax

def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)                
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()