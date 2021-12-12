#### Debuggable Deep Networks
___

**Idea:** Make the linear layer sparse by retraining a linear classifier on learned representation using elastic net regularization. A sparse linear model allows us to reason about the network’s decisions in terms of a significantly smaller set of deep features. When used in tandem with off-the-shelf feature interpretation methods, the end result is a simplified description of how the network makes predictions.

---

**Motivation**

As machine learning models find wide-spread application, there is a growing demand for interpretability: access to tools that help people see why the model made its decision. In many of its application, users are asked to trust a model to help them make decisions. A doctor will certainly not operate on a patient simply because “the model said so.” Even in lower-stakes situations, such as when choosing a movie to watch from Netflix, a certain measure of trust is required before we surrender hours of our time based on a model. Despite the fact that many machine learning models are black boxes, understanding the rationale behind the model’s predictions would certainly help users decide when to trust or not to trust their predictions. Existing work on deep network interpretability has largely approached this problem from two perspectives.

- The first one seeks to uncover the concepts associated with specific neurons in the network, for example through visualization or semantic labeling

- The second aims to explain model decisions on a per-example basis, using techniques such as local surrogates and saliency maps

While both families of approaches can improve model undderstanding at. a local level, recent work has argued that such localized explanations can lead to misleading conclusions about the model’s overall decision process. As a result, it is often challenging to flag a model’s failure modes or evaluate corrective interventions without in-depth problem-specific studies.

**Solution approach**

The solution would be to find a sparse linear layer. And instead of looking at the top-K features of the linear layer, the proposed solution re-train a linear classifier on learned representations using elastic net regularization. So our objective is minimizing the following function:

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxbpd69r36j31fw06ijs2.jpg" alt="image-20211212134337784" width="300" />

where f(xᵢ)ᵀ represents the pre-trained deep features and βs the elastic net regularization. 

____


**Repo structure**

`language` - all language related experiments and helper functions 

`vision` - all vision related experiments, helper functions and visualization tools



**Pipeline**

1. Load the dataset and its train/val/test loaders

2. Define the model
   The architecture of the model can be anything with the requirement that it has to implement a custom forward function that supports the following structure:

   ```python
   def forward(x: tensor, with_latent: bool, fake_relu: bool, no_relu: bool):
       """
       Parameters
       ----------
       x: tensor
         The input tensor
       with_latent: bool
         If this option is given, forward should return the output of the second-last layer along
         with the logits
       fake_relu: bool
         If this option is given, replace the ReLU just after the second-last layer with a
         custom_modules.FakeReLUM, which is a ReLU on the forwards pass and identity on the
         backwards pass
       no_relu: bool
         If this option is given, then with_latent should return the pre-ReLU activations of the
         second-last layer
       """
       pass
   ```

3. If the model is not pretrained, setup the training arguments and start training; otherwise skip to step 4

4. Restore the model and its weights

5. Compute the features
   Save the output (latent) from the `forward` function together with its true labels for all the samples in the dataset

6. Calculate the regularization path
   A regularization path is a plot of all coefficients values against the values of betas. The path algorithm for the elastic net calculates the regularization path where sparsity ranges the entire spectrum from the trivial zero model (β = 0) to completely dense. The solver of choice is `glm_saga` ([link](https://github.com/MadryLab/glm_saga)) which requires the following arguments:

   - `linear`: a PyTorch `nn.Linear` module which the solver initializes from (initialize this to zero)
   - `loader`: a dataloader which returns examples in the form `(X,y,i)` where `X` is a batch of features, `y` is a batch of labels, and `i` is a batch of indices which uniquely identify each example. *Important: the features must be normalized (zero mean and unit variance) and the index is necessary for the solver*. Optionally, the dataloader can also return `(X,y,i,w)` where `w` is the sample weight.
   - `max_lr`: the starting learning rate to use for the SAGA solver at the starting regularization
   - `nepochs`: the maximum number of epochs to run the SAGA solver for each step of regularization
   - `alpha`: a hyperparameter for elastic net regularization which controls the tradeoff between L1 and L2 regularization (typically taken to be 0.8 or 0.99). `alpha=1` corresponds to only L1 regularization, whereas `alpha=0` corresponds to only L2 regularization.
   
   The solver uses a mini-batch derivative of the SAGA algorithm (a class of a variance reduced proximal gradient methods)
   
   *Stopping criterias*: Two stopping criteria are used one that terminates when the change in the estimated coefficients is small and the other one stops when the training loss has not improved by more than ε_tol for more than T epochs for some T, which is called the lookbehind stopping criteria.
   
6. Select a single sparse model. 
   The elastic net yields a sequence of linear models—with varying accuracy and sparsity (the regularization path). For both vision and NLP tasks, a validation set is used to identify the sparsest decision layer, whose accuracy is no more than 5% lower on the validation set, compared to the best performing decision layer. 

6. Visualizations
   
   Traditionally, LIME is used to obtain instance-specific explanations—i.e., to identify the superpixels in a given test image that are most responsible for the model’s prediction. However in the current setting a global understanding of deep features, independent of specific test examples is needed. To accomplish that, 
   	(1) Test images are ranked based on how strongly they activate the feature of interest and then the top-k are 	selected as the most prototypical examples for positive/negative)activation of the feature
   	(2) LIME is run on each of these examples to identify relevant superpixels which involves performing linear regression to map image superpixels to the activation of the deep feature (rather than the probability of a specific class as is typical).

___

#### Evaluations


**Vision**

Sparsity allows to get sparse models without much accuracy loss.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxbpd9qxnrj30le0h4753.jpg" alt="image-20211212151723071" width="350"  />

For example, on CIFAR-10, around 15 feature per class suffices to represent the classes, and the tradeoff is 98% sparsity with only 5% loss in accuracy. 

<img src="https://lh4.googleusercontent.com/1shSEuNcVAYur8oGQhhuWoY3iMtDFN2nTtVk1J8eqHn3O1nsp5xPBPrOv-HlTEcwPx6IucchYntCGJdLcg4ACmIqP4RJ_nDk_1hTPyiZLpxEICHbXQC3-jkYvAbadSHUJtITDcI8B-Bf" alt="img" width="400"/>



**Language**

SST word clouds visualizing the positive and negative activations for the top features of the dense and sparse decision layer

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxbpday366j317s082myw.jpg" alt="image-20211212151523011" width="600" />

In some language models such as SST and plotted below, sparsity can also increase the performance.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxbpd7l34bj30l00gkjs1.jpg" alt="image-20211212151523011" width="350" />

____

#### Reference

[1]Eric Wong, Shibani Santurkar and Aleksander Madry, . "Leveraging Sparse Linear Layers for Debuggable Deep Networks". CoRR abs/2105.04857. (2021). <br>
[2]Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. “Deep inside convolutional networks: Visualising image classification models and saliency maps”. In: arXiv preprint arXiv:1312.6034 (2013).<br>
[3]Alvin Wan et al. “NBDT: neural-backed decision trees”. In: arXiv preprint arXiv:2004.00221(2020).<br>
[4]Jerome Friedman, Trevor Hastie, and Rob Tibshirani. “Regularization paths for generalized linear models via coordinate descent”. In: Journal of statistical software (2010).<br>
[5]Rie Johnson and Tong Zhang. “Accelerating stochastic gradient descent using predictive variance reduction”. In: Advances in neural information processing systems 26 (2013), pp. 315–323.<br>
[6]Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. “SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives”. In: Advances in neural information processing systems (NeurIPS). 2014<br>
