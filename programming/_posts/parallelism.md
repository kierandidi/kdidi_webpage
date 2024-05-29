---
layout: post
title: Parallelism and other GPU optimisations
image: /assets/img/blog/terminal.png
accent_image: 
  background: url('/assets/img/blog/pjs.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  How to make that data fit into your GPU
invert_sidebar: true
categories: programming
#tags:       [programming]
---

# Parallelism and other GPU optimisations

Intro and 3 categories

## Attention background

## Category 1: make exact attention faster


## Flash Attention

### Tiling

in matrix multiplication, each of the n^2 outputs uses 2n inputs, with every input being used n times and therefore n times read from main memory (i.e. DRAM). This is inefficient, we want to try and reuse params. Similar in many patterns


### attention

Attention as classification

q: d-dimensional input activations
K: cxd final layer
l = qK^T: logits
class probabilities after softmax: p_i = \exp (\frac{}{}) 

in attention, q and w come from linear operations
qq+t = Qx_t, k_s = Kx_s

- which rows s of v_s=Vx_s should we pick? Classification problem

- multi head attention is embarassingly parallel since heads operate fully indepently! Although in original paper they describe it as a research contribution that helps attention via different representation subspaces which is not possible in one bigger head due to averaging inhibiting that.

### V1

stable softmax etc, log space


May 2022, Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

efficient memory attention paper from google 

### V2

### Adding the feedforward layers to the optimizations

Aug 2023 Hao Liu et al., [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370)

## Category 2: make exact attention more scalable

### Pipeline Parallelism

### Tensor Parallelism

### Ring Attention: Same concept, different devices

Ring attention -> Flash attention
Sequence parallelism -> Blockwise parallel paper

send and receive KV segments while computation happens
- if sequences are long enough (enough compute, e.g. in paper seq length 6000) we have zero overhead
since we can overlap computation and communication


Nov 2023, Hao Liu et al., [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

pseudo algorithm


Feb 2024, Hao Liu et al., [World Models on Million-Length Video and Language With RingAttention](https://arxiv.org/abs/2402.08268s)


### ScaleFold: DAP

### Striped attention and causal masking

Nov 2023, Brandon et al., [Striped Attention: Faster Ring Attention for Causal Transformers](https://arxiv.org/abs/2311.09431)

causal masking: no attention into future, mask should be -inf so softmax output is 0

dot(Q_i, K_j) i  i <= j else -inf
-> compute this part on the fly in the kernel if part of the block is part of mask;
flash attention skils completely masked key blocks

problem for ring attention: devices idle when causally masking, ouput is 0 if query_index < key_ndex

=> in ring attention, slowest ring host determines pace

solution: striped attention (reorder QKV via stripe permutation)

- computations are almost perfectly distributed, no idle devices anymore
- further improve performance by dropping first query and last key if host_id < round; then the 
computation matrix is 1x1 smaller, and can still be computed via standard flash attention kernel
for the block since the mask is still a valid causal mask

### Flash Decoding

Flash Attention is not optimal for long-context inference (ring attention also not since it is simly flash attention multi device)

parallelises across blocks of queries and batch size only,
deoes not manage to keep GPU occupied fully during token-by-token decoding

flash decoding: this time again split keys and values across devices, but instead of communication/ring
we just do a reduction step to get the proper inference output

Flash vs speculative decoding:





## Category 3: Approximate attention


## Category 4: PerceiverIO: decoupling attention from sequence length

## Category 5: do away with attention all together

Mamba, etc