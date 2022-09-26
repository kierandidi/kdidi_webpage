---
layout: post
title: Keeping up with the literature - Zotero and Obsidian
image: /assets/img/blog/research_papers_pic.jpg
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  A practical guide to set-up Zotero and Obsidian for your perfect Zettelkasten system
invert_sidebar: true
---

# Keeping up with the literature - Zotero and Obsidian

Reading papers is something everyone in research has to do and still everyone does it in a different way. Some scribble down their notes with feather and ink in their leather-bound journal while hearing to the latest dark academia playlist on Spotify, while others highlight the hell out of a manuscript until you only notice the lines where no marker left its trace of neon-yellow destruction. I tried some of them, but converged to one which uses two free tools that work well together: Zotero for reference management and Obsidian for note-taking. In this post, I want to document how I configured my setup and how I typically go about reading a paper. Hopefully it is of use to anyone, even if that someone may be me in a month who has forgotten what he did just a few weeks ago.

* toc
{:toc}


## My workflow

First I will start describing how my typical workflow looks like so you can decide if it is something for you; afterwards I will show you how to set it up.

I really liked [this article in Nature](https://www.nature.com/articles/d41586-022-01878-7?utm_source=Nature+Briefing&utm_campaign=ce71eee966-briefing-dy-20220711&utm_medium=email&utm_term=0_c9dfd39373-ce71eee966-42664211) sharing some advice on the overall process of keeping up with the literature. In this post, I will focus mostly on the second step (managing your papers), but at this stage will shortly share how I go about the others as well:

1. Find literature: for this I mainly use different Slack channels I am part of as well as Twitter. In addition, I often consult [ResearchRabbit](https://www.researchrabbit.ai/) when exploring an unfamiliar topic; it helps you get an overview of a subject and find related literature to the articles you feed in and has a nice integration with Zotero.

2. Manage: that is where Zotero comes into play. I just use the browser plugin to add it into one of my various folders and save it for later. Folders are nice for a very broad classification of your papers, but I also use tags quite often in order to get a bit more granular and classify notes that fit multiple categories.

3. Read: well, this is probably the hardest part. I try to make fixed slots for reading in order to keep up with it. My biggest problem is getting caught up in the rabbithole of opening more and more cited papers until it gets hard to click the individual tabs anymore, so I restrict the number of open papers to three. I normally make a first read in Zotero and annotate with colors (more on that later), then export this annotations to Obsidian and give it a second read during which I address the annotations I had in the first one and make notes in Obsidian.

4. Organize: this should hopefully get easier after reading this post. Zotero and Obsidian do a good job in helping you keep your database clean, but in the end it is up to you not to clog it with piles of unread papers. My best advice here is to resist adding every paper you find to your citation manager, but be realistic in what you will be able to read and what not.

With that out of the way, let us talk about the setup of this workflow!

## Installing Zotero

You can just install Zotero from [their website](https://www.zotero.org/download/). I also recommend installing the Browser plugin for Chrome/Firefox since it makes adding papers to Zotero a lot easier.

Then you will need to install the BetterBibTex addon from [GitHub](https://github.com/retorquere/zotero-better-bibtex/releases/tag/v6.7.23) to make it integrate seamlessly with Obsidian. After downloading the .xpi file, go to Zotero and follow [these installation instructions](https://retorque.re/zotero-better-bibtex/installation/).

## Installing Obsidian

Similar to Zotero, just install Obsidian via [their website](https://obsidian.md/download). Once you have this, there are a million options on how to customize it and make it work with Zotero. I linked some example workflows in my [setup post](), but will only present the one I settled on here.

I use different Obsidian plugins, mainly [Zotero Integration](https://electricarchaeology.ca/2022/07/12/obsidian-zotero-integration-plugin/) for linking it to Zotero, Dataview for quering your notes and 

In case you do not want to customize the whole thing yourself and want a solution that works straight out of the box, I recommend [this vault](https://github.com/erazlogo/obsidian-history-vault) on GitHub (Vault being the Obsidian word for a folder containing all your notes, configurations etc). It is created by [Elena Razlogova](http://elenarazlogova.org/) who provides [amazing instructions](https://publish.obsidian.md/history-notes/01+Notetaking+for+Historians) on how to use it. While it is intended for historians, I found it easy to adapt for research in computer science and natural sciences. 

The easiest way to get it is downloading the zip file from the GitHub link above, extract it, rename it to whatever you like (in my case `obsidian-knowledgebase `) and open it with Obsidian as a new vault. This will open the vault with all the configurations, plugins and settings described on her website and including some example notes on how to use it!

Here the highlights that I use (the vault has way more configurations/options in it, so check out Elena's post linked above):

- Simply start a literature note by pressing `Cmd + R` and select the Zotero entry you want to use. It automatically gets metadata, annotations etc and creates a template for your literature note.

- It automatically extracts your color highlights into text boxes in your literature note. I configured it in the following way for me (inspired by [this post](https://electricarchaeology.ca/2022/07/12/obsidian-zotero-integration-plugin/)):

  - Yellow: used for important parts, only color that does not get exported to note.
  - Red: disagree with author
  - Green: agree with author
  - Blue: definition/concept
  - purple: unclear

To change this, you can go in your vault to meta > zotero > research note.

## Closing thoughts

At first it took me a bit of time to get used to this new way of working with the literature, but in retrospect it helped me immensely to organise my thoughts and feel less overwhelmed when working through new research papers. I hope it helps you as well!

## Credits

<span>Photo by <a href="https://unsplash.com/@jjying?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">JJ Ying</a> on <a href="https://unsplash.com/?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

*[SERP]: Search Engine Results Page
