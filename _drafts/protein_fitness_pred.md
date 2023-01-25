---
layout: post
title: Protein Fitness Prediction
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

* toc
{:toc}

# Protein Fitness Prediction

## Alignment-based models: specialise on one family

- unable to score indels
- needs deep MSA
- needs to be trained on a family, no sharing between different models

## Protein language models: one model to rule them all

- unable to score indels
- ignore dependencies between mutations (due to masked-marginal heuristic)
- minor issue: mismatch bewteen training and test procedure: training has 15% masking, less in testing 

## Tranception

- Lin et al 2013: introduce network in network idea that is equivalent to 1x1 convolutions and is later used in Inception networks. 1x1 convolutions to reduce dimensionality in filter dimension
- Szegedy et al 2014, Inception network: use 1x1 convolutions to reduce dimensionality and reduce computational costs in the next step (also called bottleneck layer). Now, use multiple filter sizes in parallel (use same padding to keep dimensionality constant), then concatenate the results and pass this further along for next layer.






# Resources:



## Closing thoughts

A

*[SERP]: Search Engine Results Page
