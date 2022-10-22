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

*This post is intended as recommended reading for the participants of the first part of the lecture series "Python for Data Science" at Heidelberg University which was conceptualised and organised by Lukas Jarosch and me, but should be intersting to anyone who wants to start working with Python.*

Computers seem to be everwhere today: in our offices, our kitchens and more and more in our labs as well. As a scientist in the natural sciences, you have better and better tools at your disposal that generate more and more data. And while back in the day an Excel table or even a lab notebook would have been sufficient, nowadays you often need software to process your data. While there is a growing amount of no-code software available that you can use without programming yourself, programming will probably form a growing part of your day-to-day job. Therefore we lecture [this course](https://github.com/kierandidi/python_for_scientists) to get you started with that, with this post being your initial overview of what we are going to cover!
* toc
{:toc}


## Python: the swiss army knife

Many beginners who want to learn programming are often confused about which language to start with: should I learn Pyhton? R? The newest and coolest language like Julia and Rust? Or a classic like Java or C++? 

First of all, it is important to note that the choice of your first programming language is actually not hugely important. If you continue with coding you will learn more than one language anyway, and once you have mastered one language the ideas and concepts can often be easily transferred to another one. 

Nevertheless, I want to give you an intuition of what programming languages are out there and why we choose to teach you Python.

At the end of the day, programming languages are just another tool to help you get your work done, similar to programs like Excel or Word or even similar to physical tools like a hammer or a drilling machine. So, as in real life, it does not make sense to do everything with a hammer; otherwise, every problem will look like a nail to you. So you should choose a tool or a set of tools that can do multiple things for you. 

In addition, you do not want to become a craftsperson and work with complicated and specialised equipment only to fix your new picture on the wall. So, the tools you choose should not only be flexible, but also easy to handle. 

With this metaphor in mind, let's transfer these insights to coding:

First, let's talk about versatility. Although there are many ways in which you can classify programming languages, for the purpose of this post we will keep it simple: in general there are *general-purpose programming languages (GPL)* and *domain specific programming languages (DSL)*. As the names suggest, the former are languages that are used for all kind of applications, wheras the latter were designed with a specific application in mind. 
This does not mean that DSLs cannot perform calculations that GPLs can (most programming languages are [Turing-complete]() anyway), but their syntax and structure are optimised for a specific purpose, which may make it harder to adapt them for others. 
The division is quite blurry in real life, but I like to keep it in the back of my head to keep my thoughts organized. [FORTRAN]() for example was originally designed for numeric computation, and although some people used it for other purposes it stayed mostly in that area. [SQL]() is another example of a language that was designed for quering databases and is nearly exclusively used for that application.

<p align="center">
<img src="/assets/img/blog/python_intro/powerdrill.png"  width="50%" height="50%"/>
</p>

*A power drill is useful for drilling holes, but not very useful for anything else.*

While these languages might be great for their respective domain, they are not suited as a first programming language since you want to learn a bredth of applications before specialising later on something you want to work on with more focus.

Therefore, we will teach you a GPL. There are many out there, from C++ over Java to Python or Julia. So, which one to choose?

Well, now our second consideration comes into play: ease of use.

Generally, people often refer to *high-level* and *low-level languages* in this area. What they mean by that is how close the language you write is to the machine code your computer reads in the end and how many of the steps in between are abstracted away by your programming language. [Assembler](https://en.wikipedia.org/wiki/Assembly_language) is an example for a language that is nearly at machine level; that gives you a lot of power and insight into the machine, but makes it useless for day-to-day tasks.

C++ is an example for a fairly low-level language. While you can also work on a higher level with libraries that give you access to object-oriented programming and other abstractions, you can still mess up your programs by playing around with low-level constructs such as [pointers](https://hackaday.com/2018/04/04/the-basics-and-pitfalls-of-pointers-in-c/). In our metaphor from above, you can think of it as a toolbox with an extensive number of complicated tools: sure, now you are not limited to one tool and are flexible, but each one of these is still quite hard to work with. 

<img src="assets/img/blog/python_intro/toolbox.jpg"  width=50% height=50%/>
*A toolbox offers you a lot of flexibility, but requires quite some expertise to be used correctly.*

C++ and Java are great languages, don't get me wrong: While learning them I learnt a lot about programming itself and the different choices you have as a programmer in how to put an abstract project specification into practice. But the course we are teaching is not primarily for programmers; it is for scientists. You do not only want to write programs, but do a lot of other things as well like doing experiments in the lab and generating the data that you will analse via your code in the end. So although I would recommend anyone interested deeper in computer science to also learn a lower-level language, in our course we will focus on Python.

Python on the other hand is what I would call the swiss army knife of programming languages. It is easy to learn, quick to prototype with and versatile in what it can be applied to.

<img src="/assets/img/blog/python_intro/swissknife.jpg"  width=50% height=50%/>
*A swiss army knife combines the best of both worlds: it is versatile and straightforward to use.*

Similar to the Swiss army knife, there are situations in which Python is not the most efficient tool. If you want to write a program doing efficient numerical calculations, Python itself will not be your saviour (but maybe one of its libraries as we will see later). In that case, a lower-level language like C++ might be more suitable. However, for the purposes of a scientist, Python is a great way into coding, both from a didactic and a practical point of view. Plus there is a large community using Python already out there, so if you get stuck there is with high probability someone out there who had the problem before you and posted a solution!

## Python libraries: your specialist tools

In this course, we will teach you two Python libraries that will come in very handy when you analyse data: Pandas and Seaborn.
### Pandas: The scissor to change data the way you like

<img src="/assets/img/blog/python_intro/scissors.png"  width=50% height=50%/>
*Similar to a pair of scissors, Pandas can slice and dice your data the way you want it, reshape it and transform it so that it fits your needs..*

### Seaborn: your magnifying glass

<img src="/assets/img/blog/python_intro/lupe.png"  width=50% height=50%/>
*Like a magnifying glass, seaborn allows you to see things in your data that you cannot see by just staring it at, and it allows you to show these insights to others.*

## Notebooks: a quick way to get started

<img src="/assets/img/blog/python_intro/pencil.png"  width=50% height=50%/>
*Notebooks help you to get a quick draft of your program into code, similar to how a pencil lets you quickly draft something on paper which can be refined afterwards.*

## Visual Studio Code: an editor you will learn to love

<img src="/assets/img/blog/python_intro/workbench.png"  width=50% height=50%/>
*All your tools at the right place: VS Code is your workbench, making it easy to access everything that you need and navigate between different tasks.*
## GitHub: collaboration is key

<img src="/assets/img/blog/python_intro/github.png"  width=50% height=50%/>
*Coding is teamwork, and GitHub helps you discuss ideas with others and show your work to the world.*

## StackOverflow: where you will spend most of your days

<img src="/assets/img/blog/python_intro/stackoverflow.png"  width=50% height=50%/>
*In case you are stuck on how to use a tool, nice people on StacOverflow can show you how to use it.*

## Closing thoughts




*[SERP]: Search Engine Results Page
