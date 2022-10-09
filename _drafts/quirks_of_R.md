---
layout: post
title: The quirks of R
image: /assets/img/blog/pythonjs.jpg
accent_image: 
  background: url('/assets/img/blog/pjs.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Destructuring can simplify your code and make it more readable
invert_sidebar: true
categories: programming
#tags:       [programming]
---

# The quirks of R
I have recently started dipping my feet into web development for an internship project where we want to expose a cloud computing application to users via an user-friendly website. And although Python in Web Development is making big steps with tools like [Gradio](https://gradio.app/) for ML web interfaces and [PyScript](https://pyscript.net/) allowing you to run Python in your browser, JS is still the language powering most web applications. 
While digging deeper into it, I found that it does things differenly than Python at some points and very similarly at others, just using different names for the same thing. Here, I want to showcase this with one particular example: Destructuring.


* toc
{:toc}


## Replacement functions


Let's look at an example:

~~~r
#define a list containing the leaderboard of CASP14 (2020)
leaderboard = ['AlphaFold2', 'BAKER', 'BAKER-experimental', 'FEIG-R2', 'Zhang', 'tFold_human']
#unpack a list into its components
first, second, third, fourth, fifth, sixth = leaderboard
print(fifth)
# > 'Zhang'
~~~






## Closing thoughts

I cannot help but feel like with the recent ES updates developers from other languages (such as Python) should be convinced to use JS due to the lower barrier of entry. While the rest operator is from my point of view a good result of this effort, there are others that are way more controversial, such as the addition of the [class syntax](https://everyday.codes/javascript/please-stop-using-classes-in-javascript/). Maybe I will cover this one in a future post.
Anyway, I hope that this post helped to clear up some confusion around the different destructuring options in Python and JS!

*[SERP]: Search Engine Results Page
