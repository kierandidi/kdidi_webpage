---
layout: post
title: Bioinformatics and the Sequencing Revolution
image: /assets/img/blog/dna_blue.jpg
accent_image: 
  background: url('/assets/img/blog/pjs.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  A hitparade of the algorithms that make modern sequencing work 
invert_sidebar: true
categories: programming
#tags:       [programming]
---

# Bioinformatics and the Sequencing Revolution

The last twenty years we witnessed a [revolution in sequencing](https://www.nature.com/immersive/d42859-020-00099-0/index.html). While sequencing the human genome was an international major project with tons of research groups involved in the late 90s and early 2000s, now the cost for the same feat [is below $1000](https://ourworldindata.org/grapher/cost-of-sequencing-a-full-human-genome). This is partly due to the technological innovations in sequencing machines and techniques (nowadays commonly referred to as *next-generation sequencing (NGS)*) that enabled cheaper and faster sequencing which gave rise to enormous amounts of genomic data. But what is often overlooked is the crucial part that bioinformatic algorithms played in this development; without innovations on the software front, this flood of data would not have lead to any actual insights. So in this post, I want to take you on a journey through the land of bioinformatic algorithms for sequencing.

- from sequence to kmers easy, but the other way around is hard!
- problem: repeats! Make it harder to look ahead
- from genome path to sequence is easy again, but many possible genome paths!

sequence alignment vs sequence assemblers

* toc
{:toc}

## Sequence Alignment vs Sequence Assembly

First of, why do we need bioinformatic algorithms for sequencing? Well, most of next-generation sequencing is currently performed using short-read sequencing (often referred to as *Solexa sequencing* or *Illumina sequencing*). These methods cannot read a human genome (3 billion base pairs) in one go, but produce millions of so-called 'reads' which are 100-300 bp in size. The questions then becomes: How do we stitch these tiny pieces together in order to get the whole sequence?

This problem can be harder or easier, depending on what information is available to you. Sequencing a human genome nowadays is way easier than back in the day of the Human Genome Project since 1. we know the human genome today and 2. newly sequenced human genomes are very similar to the ones we have and just vary at certain positions. Since we can use known human sequences as a reference, the problem reduces to aligning the sequencing reads to the right position in the reference genome. This process is called *sequence alignment*.

If you do not have any information about how the genome you're looking for looks like, well then tough luck: your only option is sticking all these tiny reads together in order to produce the final genome. This turns out be a very hard problem - due to reasons we will discuss in detail later - and is called *sequence assembly*. Nevertheless, bioinformaticians came up with quite clever ideas from graph theory to solve this problem.

In the remainder of this post, I will discuss the two problems and solutions to it in turn.

## Sequence Alignment - Where did that piece come from?



### Old school: Hashing algorithms

### New kid: Burrows-Wheeler transform

### Case study 1: How bowtie works

### Case study 2: how bzip works

## Sequence Assembly - Jigsaws for the Scientist

Harder beast, analogy of big jigsaw and problems with DNA

### Old school: Greedy algorithm assemblers

### New kid: Graph method assemblers

#### What is a De Bruijin graph? And why do we care?

### Case study 3: How Velvet works

##


~~~python
#

~~~



## Closing thoughts

A

*[SERP]: Search Engine Results Page
