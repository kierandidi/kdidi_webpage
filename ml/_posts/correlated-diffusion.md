---
layout: post
title: Correlated diffusion
image: /assets/img/blog/correlated_diffusion/corr_diffusion.png
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  How to help your model learn
invert_sidebar: true
categories: ml
---

# Correlated Diffusion

Diffusion models rock the ML stage right now. No matter if image generation or protein design, they unlock new capabilities in generative modelling.

In this blog post, I want to talk about one approach which can be leveraged to improve this process in the case of prior knowledge of correlations in the data. This blog is mainly inspired by the [Chroma paper](https://www.biorxiv.org/content/10.1101/2022.12.01.518682v1) which leverages these ideas for protein design.


## Why care about correlations?

We humans are great pattern matching machines, recognising correlations in the world around us and using these to plan and act. If you think about the structure of these correlations, they often occur on different scales: some are very simple (images and what we see around us often are quite smooth, i.e. things nearby often have a similar color), others are very complex (imagine what correlations your body has stored to tell different faces apart or distinguish different letters from another). 

In machine learning, we aim to build algorithms to learn these correlations in order to tell us something about our world (for example classifying digits by learning the correlations between the pixels in the different digit classes) and generate new things (in this example new digits or images). However, we are often limited in the resources we can give to our model in order to learn these correlations. In cases like digits, this is fine because the correlations present are quite simple to pick up, but in case of more complex systems like three-dimensional objects there are man things to learn for a model, from simple to complex correlations.

Just think about proteins: an easy correlation is that all the amino acids present will order in a line; a more complex one will be how this line then folds in order to give the final structure (the problem AlphaFold2 solved).

## Diffusion models don't care

However, current generative models often do not care about this prior knowledge we have and try to learn everythin from scratch. For example, in the case of diffusion models for proteins, we try to reverse a diffusion process that destroys all correlation in the data and transforms our data to isotropic, standard normally distributed noise. Our model then has to learn to reconstruct proteins from this noise, not only learning how proteins chains fold up but also that proteins exists as a chain in the first place!

Our model therefore has to learn not only the complex dependencies that make a protein a protein, but also simple ones like the chain constraint. It would be great if we could change that and instead focus the learning efforts of the model to the complex dependencies we actually care about.

## Transforming our diffusion process

So, how can we go about this? Let's see what is already out there in the wild.

Stable Diffusion and other open-source image generation models became efficient and affordable by an idea called *latent-space diffusion*. Here, the denoising model is sandwiched by an encoder-decoder architecture that learns to transform the data into a lower-dimensional coordinate system in which the diffusion process can be performed more efficiently (see [this video](https://www.youtube.com/watch?v=J87hffSMB60) for a more detailed explanation). 

A group from MIT worked on so-called [Subspace diffusion model](https://arxiv.org/abs/2205.01490) which aims to make the diffusion process more efficient by transforming the diffusion process to subspaces via projecting them to these subspaces during the diffusion process. This again makes it computationally more efficient and allows the model to focus its learning efforts on the important correlations in the data.

The main idea of correlated diffusion is to incorporate prior information via the covariance structure of the noise process, similar in spirit to the two ideas above. For this, one again needs to learn some kind of transformation in order to make it more 

## Whitening transformations

How could we do that? Well, we can try to come up with something called correlated diffusion. 


## Using correlations to improve learning of proteins

Proteins are incredible things, powering everything from how you move and act to how you think and perceive the world. They do this with an amazingly simple building plan: (nearly) all proteins are linear chains consisting of amino acids. 

Now imagine you want 

$$
\begin{aligned}
    q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \textbf{I}) \hspace{10px} q(x_{1:T} = \prod^T_{t=1} q(x_t | x_{t_1}))
\end{aligned}
$$

Unser Datenpunkt $$x_0$$ verliert so seine erkennbaren Eigenschaften wenn $$t$$ größer wird. Wenn $$T \to \infty$$ ist $$x_T$$ equivalent zur isotropen Normalverteilung.

![diffusion_process](/assets/img/blog/diffusion_models/diffusion_process.png)

Fig. 2. Die Markovkette des *forward (reverse) diffusion process*, in dem eine Stichprobe durch langsames Hinzufügen/Entfernen von Rauschen erzeugt wird. Quelle: [Ho et al. 2020](https://arxiv.org/abs/2006.11239) mit einigen zusätzlichen Anmerkungen.
{:.figcaption}

Eine nützliche Eigenschaft dieses Prozesses ist dass wir $$x_t$$ zu einem beliebigen Zeitpunkt $$t$$ in geschlossener Form samplen können, und zwar mithilfe eines [Reparametrisierungs-Tricks](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick). Sei $$\alpha_t = 1 - \beta_t$$ und $$\overline{\alpha_t} = \prod^t_{i=1} \alpha_i$$:

$$
\begin{aligned}%!!15
    x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}; \hspace{10px} \epsilon_{t-1}, \epsilon_{t-2}, ... \sim \mathcal{n}(0,\textbf{I}) \\[1em]
        &= \sqrt{\alpha_t \alpha_{t-1}}x_{t-2}  + \sqrt{1-\alpha_t \alpha_{t-1}} \overline{\epsilon_{t-2}} (*) \\[1em]
        &= ... \\[1em]
        &= \sqrt{\overline{\alpha_t}}x_0 + \sqrt{1-\overline{\alpha_t}}\epsilon \\[1em]

    q(x_t | x_{0}) = \mathcal{N}(x_t; \sqrt{\overline{\alpha_t}} x_{0}, (1-\overline{\alpha_t}) \textbf{I})
\end{aligned}
$$



$$ p_{\theta}(x_{0:T}) = p(x_T) \prod^T_{t=1} p_{\theta}(x_{t-1} | x_{t}) \hspace{10px} p_{\theta}(x_{t-1} | x_{t}) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t,t))$$


$$
\begin{aligned}%!!15
    x_t &=  \\[1em]
        &=  \\[1em]
        &=  \\[1em]
        &=  \\[1em]

    q(x_t \vert x_{0}) = \mathcal{N}(x_t; \sqrt{\overline{\alpha_t}} x_{0}, (1-\overline{\alpha_t}) \textbf{I})
\end{aligned}
$$




## Credits




<span>Photo by <a href="https://unsplash.com/@jjying?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">JJ Ying</a> on <a href="https://unsplash.com/?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

*[SERP]: Search Engine Results Page
