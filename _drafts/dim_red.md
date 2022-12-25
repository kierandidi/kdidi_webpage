---
layout: post
title: Dim. red.
image: /assets/img/blog/pythonjs.jpg
accent_image: 
  background: url('/assets/img/blog/pjs.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  How to go about learning Julia as a Python geek
invert_sidebar: true
categories: programming
#tags:       [programming]
---

# Dim. Red.

- contrastive learning can be seen as a pretext task (like jigsaw puzzles, inpainting, or context prediction) which are often used for pretraining for later transfer learning in semi-supervised learning. They often focus on common sense, in computer vision visually (inpainting, rotation estimation and so on).

- pretext tasks have the problem that we do not know how well this one generalizes to the downstream task. Constrastive learning to the rescue! Should be invariant to noise factors (color lighting etc) and represent how the images relate to each other. 

- build a model where different views (augmentations) of a data instance are close to each other in feature space (attractive forces) and views of other data instances are far away in feature space (repulsive forces).

- we want to learn an encoder that yields a high score for positive and a low score for negative pairs (given a chosen similarity/scoring function like cosine similarity).

- ex. loss: multi-class CE loss function with scoring function inside, called **InfoNCE loss**; its negative is a lower bound on the mutual information between the data instance and its views. We need a large negative set for this to work! (Oord Li 2018 Predictive coding repr. learning)

- design choices:
 - scoring function (like cosine similarity)
 - how to choose the examples, especially the negative ones (from different data instances, further away in one data instance, ...)
 - how to we do the augmentations (here we can combine a lot of augmentations, important for getting methods to work!)

 - SimCLR well performing framework running siamese network. Projection head only required for training, here we can play with the dimensionality of this. Problem: larger batch sizes needed, needs huge compute clusters, not practical on non-Google scale

 - to alleviate problem: **momentum contrast** used (fan et al CVPR 2020): here keep running queue (dictionary) of keys for negative samples (ring buffer of minibatches). other problems crop up like inconsistency of dictionary since encoder gets updated, for that we use moment-based update schedule. with this, MoCo2 gives same performance as SimCLR with smaller batch sizes. 

 - new method: **Barlow Twins**, information-theory-inspired approach that tries to make the task even simpler by reducing redundancy between neurons (not looking at samples but only at neurons!). Features/neurons should be invariant to data augmentations (attractive forces) but independ of the other features/neurons to remove redundancy and maximize the entropy of the representation (repulsive forces) (Zbontar LeCun 2021 redundancy reduction).
 Do this by calculating empirical cross-corr. matrix between features and encourage it to be the identity matrix (completely independent features). For this we do not need any negative samples like in classical contrastive learning (SimCLR and MoCo), very easy to implement and performs SOTA!

 **Word2Vec**

 - Efficient Estimation of Word Repre, Mikolov et al. Introduced skip-bgram and continous bag-of-words

 - next word prediction via NNLM (Bengio 2003) has problem that it scales via dominating term H x V, where H is hidden layer size and V size of vocabulary (too large to be practical, caused by softmax in final layer that sums over whole vocabulary). We want to avoid it somehow!

 - authors propose log-linear models, generate better word embeddings. They remove hidden layer.

 - they break the word-prediction problem into two subproblems: first generate word embeddings with context from before and after (continuous bag-of-word model), then train another model to actually predict the next word.

 - in continous skip-gram model they turn it around: predict surronding words by center word

-second paper: distributed representations of words and phrases and their composibitionality.

- here they replace hierarchical softmax with **negative sampling**, new optimization objective. 

- negative sampling: simplification of NCE (Noise-contrastive estimation) that is valid since we only want to learn high-quality vector repr., so we can simplify NCE as long as the vectors keep goo quality.

- instead of predicting neigbouring words (multinomial classification problem with expensive softmax), predict of given words are neigbours (cheaper binary classification problem, we can use sigmoid instead of expensive softmax function).

- for binary classifier, we need negative samples that are not neighbours. They are sampled from a noise distribution (modified smoothed monogram distribution).

Great [series of posts](http://ruder.io/word-embeddings-1/index.html) explaining word embedding models and [connection between NCE and NEG](http://ruder.io/word-embeddings-softmax/index.html#negativesampling).



# Unsupervised non-parametric methods

- MDS: arrange points in 2D to approximate high-dimensional pairwise distances. Made in 1950s-1960s by Kruskal, Torgerson. Not really better visually then PCA (parametric linear method). Also not really scalable since we need to calculate high-dimensional distances over all data points. 
Why does MDS fail although the loss function looks reasonable? As often in deep learning, the evil foe is the *curse of dimensionality*: it is just not possible to preserve high-dimensional distances in a lower-dimensional space, but that is exactly what the MDS loss function tries to do, so that does not come out very well. This problem is often called *crowding problem* and reflects the fact that in high-dimensions most points are somehow similarly close to each other, so most distances lie in a very close interval, so the important cool structure that we want to preserve with our dimensionality reduction is not contained in the distances, but in some other quantity not captured by the MDS approach.

[A definition of the crowding problem](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=visualizing_Data_using_t-SNE#:~:text=of%20symmetric%20SNE.-,The%20Crowding%20Problem,available%20to%20accommodate%20nearby%20datepoints%22.): The "crowding problem" that are addressed in the paper is defined as: "the area of the two-dimensional map that is available to accommodate moderately distant datapoints will not be nearly large enough compared with the area available to accommodate nearby datepoints". This happens when the datapoints are distributed in a region on a high-dimensional manifold around i, and we try to model the pairwise distances from i to the datapoints in a two-dimensional map. For example, it is possible to have 11 datapoints that are mutually equidistant in a ten-dimensional manifold but it is not possible to model this faithfully in a two-dimensional map. Therefore, if the small distances can be modeled accurately in a map, most of the moderately distant datapoints will be too far away in the two-dimensional map. In SNE, this will result in very small attractive force from datapoint i to these too-distant map points. The very large number of such forces collapses together the points in the center of the map and prevents gaps from forming between the natural clusters. This phenomena, crowding problem, is not specific to SNE and can be observed in other local techniques such as Sammon mapping as well.


To illustrace this, we can sample points from a standard Gaussian in 2D, 100D and 1000D and plot the distribution of pairwise distances: 

[]

We can see that the 1000D distribution does not contain any small distances, so modeling this distribution with a 2D distribution is just not going to work; we will need another approach to tackle this problem

# Neighbour Embeddings

With these methods we forget about this distance preserving idea and focus on another metric: we want to preserve nearest neighbours instead of distances. In the graph above, this would mean that we just care about the rightmost tail of the distance distribution, so we only want to preserve the rank of distances for data points.

Idea from Hinton and Roweis 2002 (Stochastic Neighbor Embedding).

t-SNE paper: Visualsing data using t-SNE (2008), way more frequently cited which has some minor modifications of original.

# How does t-SNE work?

Loss function: KL divergence between pairwise similarities/affinitis in original (high-dimensional, p) and reduced (low-dimensional, q) space. Similarity somehow opposite of distance.

High price for putting close neighbors far away (see KL divergence), low price for putting far points close together

definition of p (high-d similarities): computes Gaussian kernel via Euclidean distance (directional affinity, not symmetric). Sigma of this gaussian kernel (kernel width) is chosen adaptively to achieve desired perplexity (effective number of neighbours, per default 30). If we are in dense region, sigma is chosen to be small in order to only have 30 effective neighbours, whereas it will be increased in sparse regions.

- not symmetric: t-SNE improved over SNE by symmetrizing and normalizing directional similarities to symmetric similarities

With this directional affinities, we have all affinities as non-zero due to Gaussian kernel. But most will be around zero due to high dimensionality, so we can set them to 0 without result changes. We can also use uniform similarities for k neighbours and zero for all other points and get similar results in most cases.

definition of q (low-d similarities): similar, but it is symmetric and normalized by construction already in the SNE paper. Normalization is big difference: wheras in p we normalize by summing over all distances to neighbours of one of the points, here in q we use all of the pairwise distances in the dataset (which makes q by default symmetric).
In SNE they used the Gaussian kernel with the distance as argument, in t-SNE they changed this to a t-distribution kernel (also a Cauchy Kernel), which decays 1/d^2 instead of exponentially and is fittingly called "heavy-tailed".

We optimize the loss via classic gradient descent. 

$$
\begin{aligned} %!!15
  \phi(x,y) &= \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right) \\[2em]
            &= \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j)            \\[2em]
            &= (x_1, \ldots, x_n)
               \left(\begin{array}{ccc}
                 \phi(e_1, e_1)  & \cdots & \phi(e_1, e_n) \\
                 \vdots          & \ddots & \vdots         \\
                 \phi(e_n, e_1)  & \cdots & \phi(e_n, e_n)
               \end{array}\right)
               \left(\begin{array}{c}
                 y_1    \\
                 \vdots \\
                 y_n
               \end{array}\right)
\end{aligned}
$$

An optional caption for a math block
{:.figcaption}





# Resources:

- [evolutionary guide to t-SNE and co](https://arize.com/blog/t-sne-vs-umap/)
- [UMAP to t-SNE connection by paper co-author](https://jlmelville.github.io/uwot/umap-for-tsne.html)
- [contrastive learning applied to t-SNE and UMAP](https://arxiv.org/pdf/2206.01816.pdf)
- [presentation covering a nice red thread](https://www.bioinformatics.babraham.ac.uk/training/10XRNASeq/Dimension%20Reduction.pdf)
- [Google post on understanding UMAP](https://pair-code.github.io/understanding-umap/)
- [Kaggle notebook covering implementations of the different techniques](https://www.kaggle.com/code/samuelcortinhas/intro-to-pca-t-sne-umap)
- [details on candidate sampling techniques](https://www.tensorflow.org/extras/candidate_sampling.pdf)



 


* toc
{:toc}

## Environments

As with other programming languages, it is always a good idea to separate your projects into different environments in order to avoid conflicting package versions and improve reproducibility.

While in Python you often use venv or conda, Julia offers its own tools for that:


~~~julia
#

~~~



## Closing thoughts

A

*[SERP]: Search Engine Results Page
