---
layout: post
title: De novo proteins and where to find them - RosettaCon 2022
image: /assets/img/blog/rosetta_logo.png
accent_image: 
  background: url('/assets/img/blog/rosetta_logo.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  About the breathtaking pace of innovation in the space and the amazing community that drives it
invert_sidebar: true
---

# De novo proteins and where to find them - RosettaCon 2022

There has been a lot happening recently in protein design, and it is easy to get lost in the daily flood of new papers and exciting ideas (for an overview of current models and approaches see [this review](https://www.biorxiv.org/content/10.1101/2022.08.31.505981v1) which includes a great [model table](https://github.com/hefeda/design_tools)). 

In such situations, it often helps to step back for a second, look at the bigger picture and chat to people about what is happening and what the future might hold. Luckily, RosettaCon was happening this August and provided a venue for exactly that: chat to people about protein design and fascinating ideas! In this post I want to higlight some presentations from the conference that I think are representative of some broader directions in the field.
* toc
{:toc}


## Scaffolding protein motifs using Deep Learning - (Baker Lab, IPD, UW)

[Scaffolding Paper Science](https://www.science.org/doi/10.1126/science.abn2100)

[Diffusion models for scaffolding problem](https://arxiv.org/abs/2206.04119)
## Manifold Sampling for function-guided antibody design - Vladimir Gligorijevic (Prescient Design)

Another exciting talk by Vladimir Gligorijevic showcased some of the work that has been going on at Prescient Design and that has been published in this [Manifold Design paper](https://www.biorxiv.org/content/10.1101/2021.12.22.473759v1.full). The general idea of this approach (as you can see from the name) builds upon the [manifold hypothesis](https://www.lcayton.com/resexam.pdf), i.e. your high-dimensional data is normally not widely spread out in these dimensions, but is often restricted to a lower-dimensional manifold. 

Since functional proteins only occupy a small fraction of overall sequence space, thinking about protein design in terms of manifold sampling sounds like a reasonable idea and is what drove the recent advances in protein language models which basically learn to generate sequences that lie on this manifold of natural sequences.

But this team tackled the problem via a different approach: they built a Denoising Auto-Encoder (DAE) that takes as input a protein sequence, perturbs it and then generates a new protein sequence from there. The cool thing about that unsupervised approach is a separate supervised function classifier that predicts the function of the newly generated sequence based on Gene Ontology (GO) terms and therefore serves as guide for the DAE to generate sequences with the desired function.

The team shows some pretty cool applications of their approach, from generating new Calcium-binding proteins to an ion transporter with a novel *all-alpha* fold (no pun intended), and that by starting of from a protein with an *all-beta* fold!  In a [follow-up paper](https://arxiv.org/abs/2205.04259) they describe an approach to only design certain regions of the sequence, enabling workflows similar to the RFDesign pipeline mentioned above.

All in all this seems like a promising way to generate very diverse sequences conditioned on function.
## Designing epitope-specific binders in silico - Possu Huang (Stanford)

Generating custom antibodies binding to a specific target is already quite a feat, but doing it not only target- but even epitope-specific would be impressive. Nothing less is what Possu Huang presented at RosettaCon. They published quite some work on protein design and specifically backbone generation over the last couple years, using diverse approaches ranging from [Generative Adversarial Networks (GANs)](https://openreview.net/forum?id=SJxnVL8YOV) to [language models](http://www.proteindesign.org/uploads/1/2/1/9/121933886/2020_madani_neurips.pdf).

In 2020 they published a preprint on Ig-VAE, a Variational Autoencoder for generating antibody structures, and published the final version in [PLOS CompBio this June](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010271). 
This model is inspired by nature using a single antibody scaffold and adapting it to the problem at hand. They wanted to do something similiar *in silico*, so they chose to create a model with three important properties:

1. rotational and translational invariance should be maintained
2. the model should be aware of torsion angles since these are very important for protein structure and function
3. the output should directly be 3D structures and not an intermediate output such as distance maps.



With this new generative model, 

## Conformational Switches - Amelia McCue (North Carolina) / Florian Praetorius & Phil Leung (Baker Lab, IPD, UW)


## Bringing de novo proteins into the clinic - Javier Castellanos (Neoleukin)

[Neoleukin](https://www.neoleukin.com/) is one of several protein design companies originating in the IPD in Seattle. Their particular focus is therapeutic design with applications in e.g. [cancer immunotherapy](https://www.sciencedirect.com/science/article/pii/S1367593120300181?via%3Dihub).

Their lead candidate NL-201 is based on [this publication in Nature](https://eorder.sheridan.com/3_0/app/orders/8675/article.php) in which they showed that this de novo protein is an effective activator of IL-2 and IL-15 agonist which means it 



## Closing thoughts

## Credits

Thanks a lot to the organizers of the APFED conference, both for creating such an amazing event and for giving me permissions to use their conference logo for this blogpost.

*[SERP]: Search Engine Results Page
