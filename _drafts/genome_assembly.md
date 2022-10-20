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

If you do not have any information about how the genome you're looking for looks like, well, then tough luck: your only option is sticking all these tiny reads back together in the correct order to produce the final genome. This turns out be a very hard problem - due to reasons we will discuss in detail later - and is called *sequence assembly*. Nevertheless, bioinformaticians came up with quite clever ideas from graph theory to solve this problem.

The distinction between the two is sometimes [point of confusion](https://biology.stackexchange.com/questions/40436/what-is-the-difference-between-sequence-alignment-and-sequence-assembly), which is understandable given that in both cases we compare similarity between sequences and how to best fit them together; that's why some people are also a bit sloppy about distinguishing the two (one can even see assembly as a kind of multiple sequence alignment and therefore a special case of sequence alignment). I try to distinguishing them by what our goal is: in sequence alignment, we most often have two sequences from **different** origin (different species, different genes, ...) and try to align them to each other. In sequence assembly, however, we have sequence fragments that originate from the **same** sequence (e.g. a genome that was sequences via NGS) and try to stitch them back together into the original sequence. The basic idea is quite similar, but due to the different goals and problems different methods have developed over the years to address the two challenges.

In the remainder of this post, I will discuss the two problems and solutions to it in turn.

## Sequence Alignment - Where did that piece come from?


### Gold standard: Dynamic programming
### Old school: Hashing algorithms

e.g soap, map, blast

problem: only works great if reads to align are of same length as the length of sequence fragements with which the hash table was built. But maybe our read is longer or there is a mismatch between two exact
matches, so we want an approach with more flexibility: entering BWT!
### New kid: Burrows-Wheeler transform

bowtie (that is where it got its name from). Originally from compression. If two sequences are similar they end up close to each other in BWT, therefore alignments easy to identify. 

Which one should you choose? Well, if your read length is the same length as the k-mer length with which your hash table was produced, then hashing can even be slightly faster than BWT-based methods. But since this is seldomly the case since our reads are by now 100-300 bp long; in these cases, the BWT-based methods are **much** faster (for example, bowtie is 200-600x faster than Soap).



### Case study 1: How bowtie works

### Case study 2: how bzip works

## Sequence Assembly - Jigsaws for the Scientist

Sequence assembly is definetely the harder beast compared to sequence alignment. I like this analogy from [Wikipedia](https://en.wikipedia.org/wiki/Sequence_assembly): 

> The problem of sequence assembly can be compared to taking many copies of a book, passing each of them through a shredder with a different cutter, and piecing the text of the book back together just by looking at the shredded pieces. Besides the obvious difficulty of this task, there are some extra practical issues: the original may have many repeated paragraphs, and some shreds may be modified during shredding to have typos. Excerpts from another book may also be added in, and some shreds may be completely unrecognizable.
{:.lead}

In our context, the shredding is equivalent to short read sequencing, and piecing the book back together is what we aim to do with sequence assemlby. Repeated paragraphs correspond do DNA repeats which are [promiscous in nature and essential for genome function](https://pubmed.ncbi.nlm.nih.gov/15921050/), and modifications in the shreds correspond to sequencing errors. To put it in a nutshell: sequence assembly is damn hard. And due to the sheer size of data we are dealing with, accurate but expensive algorithms such as dynamic programming are doomed to fail.

People have tried various ways to solve the problem, and there are by now two main algorithm types used by assemblers: greedy algorithms that aim for local optima and graph method algorithms that aim for global optima.

### Old school: Greedy algorithm assemblers

### New kid: Graph method assemblers

As mentioned above, greedy algorithms have their flaws: they struggle with larger data sets since they get stuck in local optima and cannot handle repeats very well. 

Therefore, in our current age of NGS, where large data sets are the norm, graph method assemblers became the algorithm of choice. There are again two varieties of this algorihtm class: String graph and De Bruijn graph methods. Both methods improved on greedy algorithms, but while de Bruijn graph were disregarded as exotic in the beginning, they have become the de-facto standard for sequence assembly.

#### What is a De Bruijin graph? And why do we care?



#### The Euler Path

### Case study 3: How Velvet works

1. Break read into overlapping k-mers; these are nodes in directed de Bruijn graph. Choose the k-mer size in such a way that nearly all of the possible k-mers are seen at least ones (in humans that corresponds to a k-mer length of around 26-27).
2. We do this for all reads and add them to the graph via exact matching (same as done in the hashing methods during sequence alignment). This is significantly cheaper then dynamic programming and therefore relatively quick. Once we have constructed this graph, we may have some weird-looking parts branching out from the main graph like you can see below, for example due to sequencing errors; we will fix this later.
3. We can simplify the graph by merging the unambiguous node paths together into single nodes, making the whole thing a bit cleaner.
4. We remove tips, which we define as node paths that "stick out" from the graph, i.e. stop shortly after branching from the main graph and have low coverage.
5. 


### What is actually used?

SoapDeNovo (other), SGA (String Graph Aligner)a dn AbySS (De Bruijn graph method)

all of these haploid assemblers. Sometimes problem when you want to know if two variants are on same chromosome, new developments: diploid assemblers via long-read sequencing information


~~~python
#

~~~



## Closing thoughts

A

*[SERP]: Search Engine Results Page
