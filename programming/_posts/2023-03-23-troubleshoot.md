---
layout: post
title: Annoying Errors and how to fix them
image: /assets/img/blog/bugfixes/bugfixes.jpeg
accent_image: 
  background: url('/assets/img/blog/pjs.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Demystifying some of the errors out there for my future self
invert_sidebar: true
categories: programming
#tags:       [programming]
---

# Annoying Errors and how to fix them

We all now how annoying fixing bugs can be. One of the few things that is more annoying though is facing an error you encountered before but do not remember the solution for. Therefore, this post is an ongoing mental log for some of the errors I encountered at least twice to avoid repeating the bugfix process all over again from the start.

* toc
{:toc}


## CUDA Error: `Failed to initialize NVML: Driver/library version mismatch`

I encountered this one after having some PyTorch/CUDA errors, trying to reinstall some GPU drivers and failing miserably. Fortunately, after some digging I found [this discussion on the NVIDIA forum](https://forums.developer.nvidia.com/t/failed-to-initialize-nvml-driver-library-version-mismatch/190421/2) which cleared things up.

In summary, this error is caused by your GPU having a different CUDA driver version than the one you have installed on your host machine. Sometimes you already get the version as part of the error message and can rectify it based on that. If not, follow these steps:

1. run `run sudo nvidia-bug-report.sh`
2. extract the bug report via `gzip -d nvidia-bug-report.gz`
3. Open extracted `nvidia-bug-report.log` and search for “API Mismatch”. Note down which version your client (i.e. your GPU) has.
4. run `sudo apt install nvidia-driver-470` and replace `470` with whatever version your client reported in the bug report.

## pdb debugging error: `if self.quitting: raise BdbQuit (bdb.BdbQuit)`

This one I got when I was debugging an ML program. I made a local editable install of my ML repo via `pip install -e .`, had a `breakpoint()` in my dataloader and ran my model with `WandB` logging. 

What happens is that the dataloader was then executed in the background and waited for a signal to continue, step or something else, but waited with no avail and finally quit with `BdBQuit`. After reading up on this [here](https://stackoverflow.com/questions/34914704/bdbquit-raised-when-debugging-python-with-pdb), I managed to fix it by just running the dataloader itself locally in debug mode.

*[SERP]: Search Engine Results Page
