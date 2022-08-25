---
layout: post
title: JS vs Python - Destructuring
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

# JS vs Python - Destructuring
I have recently started dipping my feet into web development for an internship project where we want to expose a cloud computing application to users via an user-friendly website. And although Python in Web Development is making big steps with tools like [Gradio](https://gradio.app/) for ML web interfaces and [PyScript](https://pyscript.net/) allowing you to run Python in your browser, JS is still the language powering most web applications. 
While digging deeper into it, I found that it does things differenly than Python at some points and very similarly at others, just using different names for the same thing. Here, I want to showcase this with one particular example: Destructuring.


* toc
{:toc}


## Basic destructuring (List/Array)
Destructuring (often called unpacking in Python circles) often comes in quite handy when working in Python. What it allows you to do, described in the most abstract way, is to take some composite object, split it into its individual components and reassign those values to new identifiers in a way that is specified by you, the programmer. Or in plain English: You take something like a list, rip it mentally apart and give the pieces new names (caution with this mental model: the original list still exists!)

Let's look at an example:

~~~python
#define a list containing the leaderboard of CASP14 (2020)
leaderboard = ['AlphaFold2', 'BAKER', 'BAKER-experimental', 'FEIG-R2', 'Zhang', 'tFold_human']
#unpack a list into its components
first, second, third, fourth, fifth, sixth = leaderboard
print(fifth)
# > 'Zhang'
~~~

We had our python list (composite object), split it into its individual components (strings) and reassigned them to new identifiers (first, second and so forth).

We can do a similar thing in JavaScript: 

~~~js
//define an array containing the leaderboard of CASP14 (2020)
const leaderboard = ['AlphaFold2', 'BAKER', 'BAKER-experimental', 'FEIG-R2', 'Zhang', 'tFold_human'];
//unpack a list into its components
const [first, second, third, fourth, fifth, sixth] = leaderbord;
console.log(fifth);
// > 'Zhang'
~~~

Notice that we need square brackets in JS to indicate the destructuring, whereas in Python we just listed our elements without any additional syntax.
Mention tuples?
We do not need to match the exact number of elements in the list, but can also take a subset if we are just interested in, say, the first three entries in the leaderboard: 

~~~python
#define a list containing the leaderboard of CASP14 (2020)
leaderboard = ['AlphaFold2', 'BAKER', 'BAKER-experimental', 'FEIG-R2', 'Zhang', 'tFold_human']
#unpack a list into its components
first, second, third = leaderboard
print(second)
# > 'BAKER'
~~~

~~~js
//define a list containing the leaderboard of CASP14 (2020)
let leaderboard = ['AlphaFold2', 'BAKER', 'BAKER-experimental', 'FEIG-R2', 'Zhang', 'tFold_human']
//unpack a list into its components
let [first, second, third] = leaderboard
console.log(second)
// > 'BAKER'
~~~

But what if this subset is not consecutive, i.e. we want the first and the last entry as separate entries and all the others still together as a list, for example? Well, here the magic begins.

## Splat (*) vs Rest (...)

In Python, we can use the splat operator (*) to indicate a variable number of arguments:

~~~python
#use the splat operator for remaining elements
first, second, third, *rest = leaderbord
print(rest)
# > ['FEIG-R2', 'Zhang', 'tFold_human']

#use splat operator for elements in between
first, *middle, last = leaderboard
print(middle)
# > ['FEIG-R2', 'Zhang', 'tFold_human']
~~~

You can see exactly what the splat operator does if you apply it to a list (such as middle):

~~~python
print(middle)
# > ['BAKER', 'BAKER-experimental', 'FEIG-R2', 'Zhang']
print(*middle)
# > BAKER BAKER-experimental FEIG-R2 Zhang
~~~

It takes a list and destructures it into the separate arguments. The cool thing about that is that it does not matter how many entries our list has, we always get our first element with first, our last element with last and collect everything in between in middle!

In JavaScript, we use the so-called rest operator (...) for that:
~~~js
// file: "unpacking2.js"

//use the splat operator for remaining elements
let [first, second, third, ...rest] = leaderboard;
console.log(rest);
// > [ 'FEIG-R2', 'Zhang', 'tFold_human' ]

//use splat operator for elements in between
let [first, ...middle, last] = leaderboard;
// > Uncaught SyntaxError: Rest element must be last element
~~~

Wait: what went wrong here? Well, JS handles things a bit differently than Python here. The rest operator does exactly what its name says: it collects the rest of the elements, therefore it has be last and cannot be used to take up the arguments in the middle.
So how do we solve the issue at hand? Well, a naive approach would be to just skip the number of elements that we want to assign to middle in case we are just interested in the first and last element: 

~~~js
//skip elements in the middle
let [first,,,,,last] = leaderboard;
console.log(last);
// > 'tFold_human'
~~~

While it works, this is clearly not a viable solution. We want last to be the last element no matter how long leaderboard is: our approach here is not very flexible. 
So, what can we do? Not much, it turns out: ES6 just does not provide this feature as standard. An alternative would be to get the last element via indexing a la ES5 syntax: 

~~~js
//skip elements in the middle
let last = leaderboard[leaderboard.length-1];
console.log(last);
// > 'tFold_human'
~~~

This does the trick, but the nice one-liner we were able to write in Python is just not possible in JavaScript.


## Collecting non-keyword arguments: Splat(*) vs Rest(...)
The above-mentioned operators are often useful if you call functions and want to give the user the opportunity to give a variable number of input parameters to the function. In Python, we use the splat operator * together with args (although this is just a convention) to do that:

~~~python
def announce(intro, *args):
  print(intro)
  for rank, arg in enumerate(args):
    print(f'Rank {rank+1}: {arg}')

announce('CASP14 results:', 'AlphaFold2')
# > CASP14 results:
# > Rank 1: AlphaFold2

announce('CASP14 results:', 'AlphaFold2', 'BAKER', 'BAKER-experimental')
# > CASP14 results:
# > Rank 1: AlphaFold2
# > Rank 2: BAKER
# > Rank 3: BAKER-experimental
~~~

As you can see, we can call the function with a varying number of arguments and it accommodates them easily.

In JS, we can use the rest operator for something similar:

~~~js
function announce(intro, ...args) {
  console.log(intro);
  for (const [rank, arg] of args.entries()) {
  console.log(`Rank ${rank+1}: ${arg}`);
 }}

announce('CASP14 results:', 'AlphaFold2');
// > CASP14 results:
// > Rank 1: AlphaFold2

announce('CASP14 results:', 'AlphaFold2', 'BAKER', 'BAKER-experimental');
// > CASP14 results:
// > Rank 1: AlphaFold2
// > Rank 2: BAKER
// > Rank 3: BAKER-experimental
~~~

Before the introduction of the rest operator in ES6, you could the arguments object (similar to sys.argv in Python), but it had some weird quirks. For example, in contrast to the JS created by the rest operator, the arguments object is not an array, but an array-like object that lacks all properties except length. It also can't be used in [arrow functions](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/Arrow_functions), whereas the rest operator works fine in all kind of functions. Therefore, many prefer the use of rest over the arguments object.

~~~js
function announce(intro) {
  console.log(arguments[0]); //intro is part of arguments as well
  for (let i=1; i<arguments.length; i++) { //start from 1 to skip intro
    console.log(`Rank ${i}: ${arguments[i]}`); 
  }}

announce('CASP14 results:', 'AlphaFold2', 'BAKER', 'BAKER-experimental');
// > CASP14 results:
// > Rank 1: AlphaFold2
// > Rank 2: BAKER
// > Rank 3: BAKER-experimental
~~~


As you can see from the code snippets above, the rest operator made JS function argument unpacking more similar to Python. However, there is another thing Python can do which JS lacks currently: unpacking of keyword arguments.

## Collecting non-keyword arguments: **kwargs

In Python we can also pass a variable number of keyword (i.e. named arguments) via dictionary unpacking ** on the parameter kwargs (again, the name is just a convention, the operator ** is what really matters):

~~~python
def announce(intro, **kwargs):
  print(intro)
  for key, value in kwargs.items():
    print(f'{key}: {value}')

announce('CASP14 results:', rank1: 'AlphaFold2')
# > CASP14 results:
# > rank1: AlphaFold2

announce('CASP14 results:', rank1='AlphaFold2', rank2='BAKER', rank3='BAKER-experimental')
# > CASP14 results:
# > rank1: AlphaFold2
# > rank2: BAKER
# > rank3: BAKER-experimental
~~~

Unfortunately, at this point in time JS does not offer a similar construct.

## Rest vs Spread Operator in JS

Something many JS learners (including me) get confused by in the beginning is that the spread and the rest operator are something different despite having the same syntax of three dots (...).

In short, the rest operator puts the rest of the individual values we as a user supply to a function into a JS array, whereas the spread operator takes an interable (like a JS array) and expands it into its individual elements. Despite having similar syntax, the two operators therefore do (confusingly) quite the opposite: the rest operator converts elements into an array, the spread operator converts an array into elements.

See the code below for an example: 

~~~js
//rest operator in function call
function rank(first, ...others) {
  console.log(others); //logs an array
  //spread operator in function body
  console.log(first, ...others); //logs individual elements
}

rank(1,2,3,4)
// > [ 2, 3, 4]
// > 1 2 3 4
~~~

Both can come in handy at different points, but giving two operators different names but the same syntax does not really help in making the code more understandable. 

## Closing thoughts

I cannot help but feel like with the recent ES updates developers from other languages (such as Python) should be convinced to use JS due to the lower barrier of entry. While the rest operator is from my point of view a good result of this effort, there are others that are way more controversial, such as the addition of the [class syntax](https://everyday.codes/javascript/please-stop-using-classes-in-javascript/). Maybe I will cover this one in a future post.
Anyway, I hope that this post helped to clear up some confusion around the different destructuring options in Python and JS!

*[SERP]: Search Engine Results Page
