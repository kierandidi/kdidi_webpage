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

Reading papers is something everyone in research has to do and still everyone does it in a different way. Some scribble down their notes with feather and ink in their leather-bound journal while listening to the latest dark academia playlist on Spotify, while others highlight the hell out of a manuscript until you only notice the lines where no marker left its trace of neon-yellow destruction. I tried some of them, but converged on one which uses two free tools that work well together: Zotero for reference management and Obsidian for note-taking. In this post, I want to document how I configured my setup and how I typically go about reading a paper. Hopefully it is of use to anyone, even if that someone may be me in a month who has forgotten what he did just a few weeks ago.

* toc
{:toc}


## My workflow

First I will start describing what my typical workflow looks like so you can decide if it is something for you; afterwards I will show you how to set it up.

I really liked [this article in Nature](https://www.nature.com/articles/d41586-022-01878-7?utm_source=Nature+Briefing&utm_campaign=ce71eee966-briefing-dy-20220711&utm_medium=email&utm_term=0_c9dfd39373-ce71eee966-42664211) sharing some advice on the overall process of keeping up with the literature. In this post, I will focus mostly on the second step (managing your papers), but at this stage will shortly share how I go about the others as well:

![paper_workflow_overview](/assets/img/blog/zotero/paper_workflow_overview.png)

1. Find literature: for this I mainly use different Slack channels I am part of as well as Twitter. In addition, I often consult [ResearchRabbit](https://www.researchrabbit.ai/) when exploring an unfamiliar topic; it helps you get an overview of a subject and find related literature to the articles you feed in and has a nice integration with Zotero.

2. Manage: that is where Zotero comes into play. I just use the browser plugin to add it into one of my various folders and save it for later. Folders are nice for a very broad classification of your papers, but I also use tags quite often in order to get a bit more granular and classify notes that fit multiple categories.

3. Read: well, this is probably the hardest part. I try to make fixed slots for reading in order to keep up with it. My biggest problem is getting caught up in the rabbithole of opening more and more cited papers until it gets hard to click the individual tabs anymore, so I restrict the number of open papers to three. I normally make a first read in Zotero and annotate with colors (more on that later), then export these annotations to Obsidian and give it a second read during which I address the annotations I had in the first one and make notes in Obsidian.

4. Organize: this should hopefully get easier after reading this post. Zotero and Obsidian do a good job in helping you keep your database clean, but in the end it is up to you not to clog it with piles of unread papers. My best advice here is to resist adding every paper you find to your citation manager, but be realistic in what you will be able to read and what not.

Now that you have a general idea of how I go about reading papers, let's drill down on how I use Zotero and Obsidian:

1. First, I find a paper online (in this example on biorxiv) and import it into the respective Zotero folder via the Chrome plugin (upper right corner).

    ![paper_workflow_1](/assets/img/blog/zotero/paper_workflow_1.png)

2. Then, I open this paper in Zotero, start reading it and highlight the bits that are relevant for me according to a color scheme (more on that later).

    ![paper_workflow_2](/assets/img/blog/zotero/paper_workflow_2.png)

3. Having finished the first read, I open Obsidian and click `Cmd + R` which opens the Zotero Integrator interface. Then, I search for the article I want to import, select it and click `Enter`.

    ![paper_workflow_3](/assets/img/blog/zotero/paper_workflow_3.png)

4. Now Obsidian created a research note with the paper title, authors etc. including my annotations in colored boxes with types that I specify during the set-up (for example blue for definitions/concepts). With this note, I go through the paper again and add my own notes to Obsidian. 

    ![paper_workflow_4](/assets/img/blog/zotero/paper_workflow_4.png)

With this workflow, I turned a paper I found in the internet into an annoted Zotero entry and a custom Obsidian note for that paper. If this kind of workflow is what you're looking for, let us talk about the setup of this workflow!

## Installing Zotero

You can just install Zotero from [their website](https://www.zotero.org/download/). I also recommend installing the Browser plugin for Chrome/Firefox since it makes adding papers to Zotero a lot easier. An alternative way would be to go to Zotero, click on the magic wand (the button named `Add Item(s) by Identifier) and paste the DOI of the article every time you want to add something to your library.

Then you will need to install the BetterBibTex addon from [GitHub](https://github.com/retorquere/zotero-better-bibtex/releases/tag/v6.7.23) to make it integrate seamlessly with Obsidian. After downloading the .xpi file from GitHub, go to Zotero and follow [these installation instructions](https://retorque.re/zotero-better-bibtex/installation/).

## Zotero Storage

By default, Zotero comes with 300 MB of online storage space for your references and associated documents. Since this is not a lot you will probably run quickly out of space. From my perspective there are three main ways to resolve this problem:

1. **Do nothing**. If you online use Zotero on one device and do not need to cross-sync your data between devices, it is enough to have all your data on your local device. In my case, however, I wanted to sync my references and papers between my laptop and tablet to enable handwritten annotations and therefore needed to find a different solution.
2. **Buy Zotero storage**. Zotero offers [different storage plans](https://www.zotero.org/storage) for a monthly fee. If you want to have your files in the Zotero storage, this can be a good option. I however wanted to work with my own cloud storage solution to make annotations on my tablet easier and therefore needed to find a different solution.
3. **Use your own cloud storage**. Zotero offers the option to use your own cloud storage solution (e.g. Dropbox, Google Drive, OneDrive) to store your files. This is the option I went for. There are great resources out there that explain how to do it, for example [this video](https://www.youtube.com/watch?v=fEChje5xgQw). I now sync my files to my university OneDrive and can access these papers from both my laptop and my tablet. On my tablet I sync between OneDrive and GoodNotes, which allows me to annotate the papers with my Apple Pencil. Once I annotated something, I just sync it back to OneDrive and it will be available on my laptop as well.

## Installing Obsidian

Similar to Zotero, just install Obsidian via [their website](https://obsidian.md/download). Once you have this, there are a million options on how to customize it and make it work with Zotero. I linked some example workflows in my [setup post](https://kdidi.netlify.app/blog/tools/2022-09-17-mac-setup-copy/), but will only present the one I settled on here.

I use different Obsidian plugins, mainly [Zotero Integration](https://electricarchaeology.ca/2022/07/12/obsidian-zotero-integration-plugin/) for linking it to Zotero, Dataview for querying your notes and Templater for creating and using note templates.

In case you do not want to customize the whole thing yourself and want a solution that works straight out of the box, I recommend [this vault](https://github.com/erazlogo/obsidian-history-vault) on GitHub (Vault being the Obsidian word for a folder containing all your notes, configurations etc). It is created by [Elena Razlogova](http://elenarazlogova.org/) who provides [amazing instructions](https://publish.obsidian.md/history-notes/01+Notetaking+for+Historians) on how to use it. While it is intended for historians, I found it easy to adapt for research in computer science and natural sciences. 

The easiest way to get it is the following:

- download the zip file from [GitHub](https://github.com/erazlogo/obsidian-history-vault)
- extract the zip file
- rename it to whatever you like (in my case `obsidian-knowledgebase`)
- open it with Obsidian as a new vault. You can do this by opening Obsidian, going to `File > Open Vault...`, choosing `Open folder as vault` and then select the downloaded folder from your Downloads folder. 

This will open the vault with all the configurations, plugins and settings described on her website and including some example notes on how to use it!

![import_vault](/assets/img/blog/zotero/import_vault.png)

Here the highlights that I use (the vault has way more configurations/options in it, so check out Elena's post linked above):

- Simply start a literature note by pressing `Cmd + R` and select the Zotero entry you want to use. It automatically gets metadata, annotations etc and creates a template for your literature note.

- It automatically extracts your color highlights into text boxes in your literature note. I configured it in the following way for me (inspired by [this post](https://electricarchaeology.ca/2022/07/12/obsidian-zotero-integration-plugin/)):

  - Yellow: used for important parts, only color that does not get exported to note.
  - Red: disagree with author
  - Green: agree with author
  - Blue: definition/concept
  - purple: unclear

To change this, you can go in your vault to `meta > zotero > research note`.

## Closing thoughts

At first it took me a bit of time to get used to this new way of working with the literature, but in retrospect it helped me immensely to organise my thoughts and feel less overwhelmed when working through new research papers. I hope it helps you as well!



*[SERP]: Search Engine Results Page
