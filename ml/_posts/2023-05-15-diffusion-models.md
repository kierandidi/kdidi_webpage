---
layout: post
title: (GER) Was sind Diffusion Models
image: /assets/img/blog/diffusion_models/diffusion_models_cover.jpg
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Was hinter dem Hype steckt
invert_sidebar: true
categories: ml
---

# (GER) Was sind Diffusion Models

(Die deutsche Version beginn unten!)


This post is a rather unusual one since it is in German. I have always been involved in making content available in other languages to allow more people to enjoy it, such as when I did translations for Khan Academy. After translating the posts on normalising flows by Eric Jang, I have the pleasure of now translating [Lily Wang's excellent post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#classifier-free-guidance) on diffusion models. I hope you enjoy it!


* toc
{:toc}

## Einführung

GANs, VAEs und Normalising Flows sind drei Typen von Machine Learning Modellen für generative Zwecke. Alle drei haben sehr erfolgreich hochqualitative Beispiele generiert, aber jede der drei Familien hat eigene Probleme. GANs sind bekannt für instabiles Training und weniger Diversität der produzierten Beispiele durch ihr Training. VAEs basieren auf einem sogenannten "surrogate loss". Normalising Flows müssen spezielle Architekturen verwenden, um reversible Transformationen zu konstruieren.

Diffusion Models sind von der "non-equilibrium" Thermodynamik inspiriert. Sie definieren eine Markov-Kette von Diffusionsschritten, um den Daten langsam zufälliges Rauschen hinzuzufügen, und lernen dann, den Diffusionsprozess umzukehren, um aus dem Rauschen gewünschte Datenproben zu konstruieren. Im Gegensatz zu VAEs oder Normalising Flows werden Diffusion Models mit einem festen Verfahren erlernt, und die latente Variable hat eine hohe Dimensionalität (dieselbe wie die Originaldaten).

![gen_model_overview.png](/assets/img/blog/diffusion_models/gen_model_overview.png)


## Was sind Diffusion Models

Es wurden mehrere diffusionsbasierte generative Modelle mit ähnlichen Ideen vorgeschlagen, darunter diffusion probabilistic models ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)), noise-conditioned score network (NCSN; [Yang & Ermon, 2019](https://arxiv.org/abs/1907.05600)), und denoising diffusion probabilistic models (DDPM; [Ho et al. 2020](https://arxiv.org/abs/2006.11239)).



## Forward Diffusion Process

Nehmen wir an, wir haben einen Datenpunkt von einer realen Datenverteilung, $$x_0 \sim q(x)$$. Dann können wir einen *forward diffusion process* definieren, in dem wir in $$T$$ Schritten kleine Mengen an Gaussian noise zu dem Datenpunkt hinzufügen und damit eine Sequenz $$x_1, ..., x_T$$ an korrumpierten (sogenannten *noised*) Datenpunkten erzeugen. Wir kontrollieren die Schrittgröße zwischen diesen Datenpunkten mit der sogenannten *variance schedule* $$\{\beta_t \in (0,1)\}^T_{t=1}$$.

$$
\begin{aligned}
    q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \textbf{I}) \hspace{10px} q(x_{1:T} = \prod^T_{t=1} q(x_t | x_{t_1}))
\end{aligned}
$$

Unser Datenpunkt $$x_0$$ verliert so seine erkennbaren Eigenschaften wenn $$t$$ größer wird. Wenn $$T \to \infty$$ ist $$x_T$$ equivalent zur isotropen Normalverteilung.

![diffusion_process](/assets/img/blog/diffusion_models/diffusion_process.png)

A caption for an image.
{:.figcaption}

Eine nützliche Eigenschaft dieses Prozesses ist dass wir $$x_t$$ zu einem beliebigen Zeitpunkt $$t$$ in geschlossener Form samplen können, und zwar mithilfe eines [Reparametrisierungs-Tricks](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick). Sei $$\alpha_t = 1 - \beta_t$$ und $$\overline{\alpha_t} = \prod^t_{i=1} \alpha_i$$:

$$
\begin{aligned}
    x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}; \hspace{10px} \epsilon_{t-1}, \epsilon_{t-2}, ... \sim \mathcal{n}(0,\textbf{I}) \\[2em]
        &= \sqrt{\alpha_t \alpha_{t-1}}x_{t-2}  + \sqrt{1-\alpha_t \alpha_{t-1}} \overline{\epsilon_{t-2}} (*) \\[2em]
        &= ... \\[2em]
        &= \sqrt{\overline{\alpha_t}}x_0 + \sqrt{1-\overline{\alpha_t}}\epsilon \\[2em]

    q(x_t | x_{0}) = \mathcal{N}(x_t; \sqrt{\overline{\alpha_t}} x_{0}, (1-\overline{\alpha_t}) \textbf{I})
$$

(*) Wenn wir zwei Normalverteilungen mit verschiedenen Varianzen kombinieren, hat die neue Normalverteilung die Summe der Varianzen als Varianz: $$\mathcal{N}(0, \sigma^2_1\textbf{I}) + \mathcal{N}(0, \sigma^2_2\textbf{I}) = \mathcal{N}(0, (\sigma^2_1 + \sigma^2_2)\textbf{I})$$. In unserem Falle ist die kombinierte Standardabwecihung $$\sqrt{(1-\alpha_t) + \alpha_t(1-\alpha_{t-1}} = \sqrt{1-\alpha_t \alpha_{t-1}}$$.

Normalerweise können wir uns größere Updateschritte erlauben wenn unsere Sample mehr Rauschen enthält, also setzen wir die variance schedule so, dass $$\beta_t$$ mit $$t$$ wächst: $$\beta_1 < \beta_2 < ... < \beta_t$$ und daher $$\overline{\alpha_1} > \overline{\alpha_2} > ... > \overline{\alpha_t}$$.

## Verbindung zu Stochastic Gradient Langevin Dynamics










## Credits

Vielen Dank an Eric Jang für ein cooles Tutorial und die Möglichkeit, es deutschen Lesern zugänglich zu machen!
Außerdem vielen Dank an Brad Saund, dessen [Blogpost](https://www.bradsaund.com/post/normalizing_flows/) ähnliche Beispiele zu den obigen in TensorFlow2 enthält!


<span>Photo by <a href="https://unsplash.com/@jjying?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">JJ Ying</a> on <a href="https://unsplash.com/?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

*[SERP]: Search Engine Results Page
