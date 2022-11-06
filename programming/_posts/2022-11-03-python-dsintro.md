---
layout: post
title: Python for Data Science - know your tools!
image: /assets/img/blog/python_intro/toolbox_front.png
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  How to get started with Python and what tools to use
invert_sidebar: true
categories: programming
---

# Python for Data Science

*This post is intended as recommended reading for the participants of the first part of the lecture series "Python for Data Science" at Heidelberg University which was conceptualised and organised by Lukas Jarosch and me, but should be interesting to anyone who wants to start working with Python.*

Computers seem to be everwhere today: in our offices, our kitchens and increasingly in our labs as well. As a scientist in the natural sciences, you have better and better tools at your disposal that generate more and more data. And while back in the day an Excel table or even a lab notebook would have been sufficient, nowadays you often need software to process your data. While there is a growing amount of no-code software available that you can use without programming yourself, programming will probably form a growing part of your day-to-day job. This is why we are holding [this course](https://github.com/kierandidi/python_for_scientists) to get you started, with this post as your initial overview of what we are going to cover!

* toc
{:toc}


## Python: the swiss army knife

Many beginners who want to learn programming are confused about which language to start with: should I learn Python? R? The newest and coolest language like Julia and Rust? Or a classic like Java or C++? 

First of all, it is important to note that the choice of your first programming language is actually not hugely important. If you continue with coding you will learn more than one language anyway, and once you have mastered one language the ideas and concepts can often be easily transferred to another. 

Nevertheless, I want to give you an intuition of what programming languages are out there and why we choose to teach you Python.

At the end of the day, programming languages are just another tool to help you get your work done, similar to programs like Excel or Word, or even physical tools like a hammer or a drilling machine. So, as in real life, it does not make sense to do everything with a hammer; otherwise, every problem will look like a nail to you. So you should choose a tool or a set of tools that can do multiple things for you. 

In addition, you do not want to become a craftsperson and work with complicated and specialised equipment only to fix your new picture on the wall. So, the tools you choose should not only be flexible, but also easy to handle. 

With this metaphor in mind, let's transfer these insights to coding:

First, let's talk about versatility. Although there are many ways in which you can classify programming languages, for the purpose of this post we will keep it simple: in general there are *general-purpose programming languages (GPL)* and *domain specific programming languages (DSL)*. As the names suggest, the former are languages that are used for all kind of applications, whereas the latter were designed with a specific application in mind. 
This does not mean that DSLs cannot perform calculations that GPLs can (most programming languages are [Turing-complete]() anyway), but their syntax and structure are optimised for a specific purpose, which may make it harder to adapt them for others. 
The division is quite blurry in real life, but I like to keep it in the back of my head to keep my thoughts organised. [FORTRAN]() for example was originally designed for numeric computation, and although some people used it for other purposes it stayed mostly in that area. [SQL]() is another example of a language that was designed for querying databases and is nearly exclusively used for that purpose.

<p align="center">
  <img src="/assets/img/blog/python_intro/powerdrill.png" width="50%" height="50%"/>
</p>

*A power drill is useful for drilling holes, but not very useful for anything else.*

While these languages might be great for their respective domain, they are not suited as a first programming language since you want to learn a breadth of applications before specialising later in something you want to work on with more focus.

Therefore, we will teach you a GPL. There are many out there, from C++ over Java to Python or Julia. So, which one to choose?

Well, now our second consideration comes into play: ease of use.

Generally, people often refer to *high-level* and *low-level languages* in this area. What they mean by that is how close the language you write is to the machine code your computer reads in the end and how many of the steps in between are abstracted away by your programming language. [Assembler](https://en.wikipedia.org/wiki/Assembly_language) is an example of a language that is nearly at machine level: it gives you a lot of power and insight into the machine, but makes it useless for day-to-day tasks.

C++ is an example for a fairly low-level language. While you can also work on a higher level with libraries that give you access to object-oriented programming and other abstractions, you can still mess up your programs by playing around with low-level constructs such as [pointers](https://hackaday.com/2018/04/04/the-basics-and-pitfalls-of-pointers-in-c/). In terms of our previous metaphor, you can think of it as a toolbox with an extensive number of complicated tools: sure, now you are not limited to one tool and are flexible, but each individual one of these is still quite difficult to work with. 

<p align="center">
  <img src="/assets/img/blog/python_intro/toolbox.jpg" width="50%" height="50%"/>
</p>

*A toolbox offers you a lot of flexibility, but requires quite some expertise to be used correctly.*

C++ and Java are great languages, don't get me wrong: while learning them I learnt a lot about programming itself and the different choices you have as a programmer in how to put an abstract project specification into practice. But the course we are teaching is not primarily for programmers; it is for scientists. You do not only want to write programs, but do a lot of other things as well like experimenting in the lab and generating the data that you will analyse via your code in the end. So although I would recommend also learning a lower-level language to anyone with a deeper interest in computer science, in our course we will focus on Python.

Python is what I would call the swiss army knife of programming languages. It is easy to learn, quick to prototype with and versatile in what it can be applied to.

<p align="center">
  <img src="/assets/img/blog/python_intro/swissknife.jpg" width="50%" height="50%"/>
</p>

*A swiss army knife combines the best of both worlds: it is versatile and straightforward to use.*

Similar to the Swiss army knife, there are situations in which Python is not the most efficient tool. If you want to write a program doing efficient numerical calculations, Python itself will not be your saviour (but maybe one of its libraries, as we will see later). In that case, a lower-level language like C++ might be more suitable. However, for the purposes of a scientist, Python is a great way into coding, both from a didactic and a practical point of view. Plus there is a large community using [Python](https://www.python.org/) already out there, so if you get stuck, there is a high probability that someone out there had the problem before you and posted a solution!

## Python libraries: your specialist tools

In this course, we will teach you two Python libraries that will come in very handy when you analyse data: Pandas and Seaborn.

### Pandas: The scissor to change data the way you like

As a scientist, you will often find yourself doing an experiment and generating large amounts of data from it that you want to analyse. Doing it by hand is impossible due to the number of data points, and tools like Excel start to freeze already when opening your data file. So, what do you do?

Enter [Pandas](https://pandas.pydata.org/), a library for data analysis in Python. With it, many of the tasks for which you would need to write your own functions in base Python (opening Excel sheets, calculating summary statistics, filtering/combining data) are just a one-liner. It comes with its own data structure called a [Pandas DataFrame](https://realpython.com/pandas-dataframe/), which you will learn a lot more about during the course. You can imagine it as a table storing your data in a convenient and efficient format.

<p align="center">
  <img src="/assets/img/blog/python_intro/scissors.png" width="50%" height="50%"/>
</p>

*Similar to a pair of scissors, Pandas can slice and dice your data the way you want it, reshape it and transform it so that it fits your needs.*

The Pandas website has some great [tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) on how to get started with Pandas and a [cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html) on how to use it for specific cases, so if you find a use case after the course that you did not encounter before or you do not remember how to handle, these are great places to start looking!

Pandas builds on another library called [NumPy](https://numpy.org/). NumPy is a library optimised for efficiency via [vectorization](https://en.wikipedia.org/wiki/Array_programming) and is often used in scientific applications. However, often you do not need all the flexibility that NumPy offers you and want a more concise and pre-structured formulation of your code. Here Pandas can shine: it sits in between the simplicity of base Python and the efficiency of NumPy.

### Seaborn: your magnifying glass

Often, transforming your data via Pandas is only the first step of your data analysis. Numbers can only show so much, and it is often more efficient to visualise your data via graphics, both for your own understanding of your data as well as for communicating your insights to others. Here [Seaborn](https://seaborn.pydata.org/) comes to your rescue: it is a data visualisation library that allows you to directly take your data from a DataFrame and plot it easily via a myriad of formats, with all kinds of styles and customisations available. It integrates very well with Pandas and makes creating graphs dead-easy. In case you are looking for inspiration, there is an [example gallery](https://seaborn.pydata.org/examples/index.html) showing plots with the corresponding code so that you can easily adapt them for your own purposes.

<p align="center">
<img src="/assets/img/blog/python_intro/lupe.png" width="50%" height="50%"/>
</p>

*Like a magnifying glass, seaborn allows you to see things in your data that you cannot see by just staring it at, and it allows you to show these insights to others.*

Similar to Pandas, Seaborn did not come from nowhere: it is based on the library [matplotlib](https://matplotlib.org/), which is the go-to library for visualisation in Python. Again, similar to NumPy it offers you a lot of flexibility, but often you will prefer readable over extremely flexible code when analysing your data. Especially the strong integration with Pandas gives you a good reason to use Seaborn. That being said, in many circumstances I find myself switching back and forth between Pandas/Seaborn and NumPy/matplotlib; since the former two are based on the latter two, using them together often works quite well!

## Visual Studio Code: an editor you will learn to love

Pandas and Seaborn and all the rest of it are great, but where do you actually write the code containing all these libraries? While you could do that just via the command line, there are way better tools available nowadays that make your life a lot easier. These are often called [integrated development environment](https://en.wikipedia.org/wiki/Integrated_development_environment)(IDE) and if you participated in the course [Data Analysis with R](https://jmbuhr.de/dataintro/) by Jannik Buhr, you already met one of these: RStudio.

While RStudio is a nice IDE, it is very R-centric in many of its design decisions. For our purposes, we want a general-purpose IDE that can act as our workbench: we bring whatever tools we want to work with (programming language, packages, data etc.) and our IDE should support our work with these. That is why we decided to teach this course using [VS Code](https://code.visualstudio.com/). It is versatile, open-source and has a massive library of extensions that make your life as a programmer easier. To get started, see [this amazing guide](https://realpython.com/python-development-visual-studio-code/) which will take you through the different steps of installing and setting up VS Code (there is also a short walkthrough-guide available on the [VS Code website](https://code.visualstudio.com/docs/python/python-tutorial)).

<p align="center">
  <img src="/assets/img/blog/python_intro/workbench.png" width="50%" height="50%"/>
</p>

*All your tools at the right place: VS Code is your workbench, making it easy to access everything that you need and navigate between different tasks.*

To learn more about the cool things you can do with VS Code (support for R, connecting to remote machines, keyboard shortcuts etc) you can have a look at [this post](https://kdidi.netlify.app/blog/tools/2022-09-17-mac-setup/) which I published some time ago. 

## Notebooks: a quick way to get started

Using the tools we mentioned so far you can create great programs for analysing your data. But how to get started? And how to document what you have done? And how to put your work into a format that is suitable for presenting at e.g. a lab meeting?

This is were noteboks come in ([Jupyter notebooks](https://jupyter.org/) for Python more specifically with the file ending `.ipynb`]). You can just power up your notebook and get started coding; it shows you the output directly in the document, no matter if it is a number, a plot or an image.

And once you finish your analysis, you can just add some text cells in which you describe your code more eloquently via [Markdown syntax](https://www.datacamp.com/tutorial/markdown-in-jupyter-notebook) than inline-comments in your code could ever do.

<p align="center">
  <img src="/assets/img/blog/python_intro/pencil.png" width="50%" height="50%"/>
</p>

*Notebooks help you to get a quick draft of your program into code, similar to how a pencil lets you quickly draft something on paper which can be refined afterwards.*

These notebooks can serve both as a quick start to some data exploration (e.g. visualising your data for quality control) and as a documentation of your work you can show to colleagues and collaborators. In fact, the lecture slides we will deliver as part of the Python course are made from notebooks!

## GitHub: collaboration is key

Notebooks are a great way to share your results once you are done with analysing your data, but what if you want to share it with your colleagues who want to work on the analysis together with you? The most straightforward way would be to just send them the `.ipynb` file which they can then use to work on the data. This however has some caveats: first of all, your colleague needs to have the exact same format of the data as you in order to make the analysis reproducible. Second, if your analysis becomes more complicated, you may want to split your analysis into different Python files and notebooks, making exchanging these more complicated. In addition, you cannot work on the analysis at the same time, since the changes you and your colleague introduce might not be compatible and therefore merging these changes into a consistent program in the end might just be impossible. Even worse, every time you change something in the code you have to send your colleague a new version of your code and vice versa, a huge waste of time and effort. That is why GitHub (and Git) were created.


<p align="center">
  <img src="/assets/img/blog/python_intro/github.png" width="50%" height="50%"/>
</p>

*Coding is teamwork, and GitHub helps you discuss ideas with others and show your work to the world.*

[GitHub](https://github.com/) is an online service for software development and [version control](https://en.wikipedia.org/wiki/Version_control). It uses Git, a system for local version control, and distributes it globally so that you can work together with people all over the world. In addition, it provides some nice features that make the software development process more structured and organised: wikis, pull requests, issues, taks management, continuous integration, basically the whole software shebang you could wish for. 

Git is a great system, though it takes some time to get used to. But I can assure you that this time will be well-spent since it is the de facto standard for publishing and sharing software with the world. In case you like gamified learning, [here](https://ohmygit.org/) is the link to a game that makes the dive into Git a bit more fun.
## StackOverflow: where you will spend most of your days

When you think about programming, you might still have the image of a hacker on your mind, sitting with his hoodie in a dark room, relentlessly maltreating his keyboard without rhyme or reason. That is far from the actual reality: you will spend most of your time looking up stuff which you either have no clue about or had a clue about at some point but lost it in the flood of other important information such as Shakira lyrics and the results of last weekend's football match. Here, [StackOverflow](https://stackoverflow.com/tour) is your friend.

<p align="center">
  <img src="/assets/img/blog/python_intro/stackoverflow.png" width="50%" height="50%"/>
</p>

*In case you are stuck on how to use a tool, nice people on StackOverflow can show you how to use it.*

On StackOverflow, you can ask questions about basically everything related to programming and will often receive high-quality answers (given that you formulated your question well). But in most cases, you won't even have to ask a question since another person had the same or a very similar problem before you and solved it with help from StackOverflow. Using this tool right is a skill that cannot be overestimated since you will spend a significant amount of time using this and other similar resources. There are [many guides out there](https://www.freecodecamp.org/news/5-steps-to-become-a-better-stack-overflow-user-4ce85711c0f9/) on how to use it best; for me, learning by doing has helped the most. And once you are advanced enough, answering questions there can be a great way to give back and stay sharp on your coding skills at the same time!

## Closing thoughts

There are many great resources out there for scientist who want to learn coding for their research, such as [this workshop webpage](https://education.molssi.org/python-scripting-biochemistry/chapters/setup.html). We hope that with our lecture series we give you both the skills to apply coding for basic tasks in your research and the enthusiam to continue learning new things in order to improve even more!


*[SERP]: Search Engine Results Page
