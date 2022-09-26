---
layout: post
title: Blender for Biochemists - A primer
image: /assets/img/blog/kras-compressed.png
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  How to use Blender as a scientist to create stunning graphics
invert_sidebar: true
categories: tools
---

# Blender for Biochemists - A primer
On Twitter I came across Brady Johnston and his [stunning Blender graphics](https://twitter.com/bradyajohnston/status/1388064679486902279?cxt=HHwWjsCqoZjessMmAAAA) for biology. Having meddled with Blender in the past, I was eager to get back into it and was amazed on the enormous progress that has been made in making Blender useful for scientific illustrations, from general concepts such as [geometry nodes](https://www.3dblendered.com/news/learning-blender/introduction-to-geometry-nodes-for-beginners-in-blender-3d/) to purpose-specific packages such as the beautiful [Molecular Nodes](https://bradyajohnston.github.io/MolecularNodes/) extension. Here, I want to give you a taste for what is possible with Blender and give you some pointers on how to get started!


* toc
{:toc}

## How to get started

First, you need to download Blender. You can get it from the [Blender website](https://www.blender.org/download/). I recommend the latest version, which is currently 3.3 (anything below 3.0 will have significantly different features/interfaces, so choose no version lower than 3.0). You can also get it from your package manager, but I recommend downloading it from the website, as it is the most up-to-date version.

## Crips and clean - create a protein snapshot

Typically you would import a protein as a PDB structure. There are different ways to do it, but the three I used were importing from PyMol, importing from ChimeraX and importing directly to Blender via the molecular nodes extension (more about that later).

Although I like PyMol as a program a lot, I would not recommend to use it in conjunction with Blender since its export format requires a lot of manual editing and cleaning in Blender (see [this video]() for more details.)
[](https://www.youtube.com/watch?v=CfkjBoOaw0g)
## Floating along - showcase your membrane proteins

[French tutorial](https://www.youtube.com/watch?v=2E4Yc5Yoa-8)

![Brady's node editor](/assets/img/blog/blender29_membrane.png)

![My node editor](/assets/img/blog/blender33_membrane.png)

New geometry nodes work with 'fields'

There are different input and output sockets in the node editor, most of them are either colored circles  or colored diamonds. 

*Circles*
- bright green: mesh
- dark green: integer
purple: vector

*Diamonds*
- purple: vector
- pink: boolean
- grey: float

If they have a dot in the middle, this means they can take a field as input.
## Spin right round - DNA in Blender


## Supercharge your workflow - molecular nodes


## 



## Closing thoughts


*[SERP]: Search Engine Results Page
