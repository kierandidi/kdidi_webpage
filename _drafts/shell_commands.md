---
layout: post
title: sed, awk & co - master the shell
image: /assets/img/blog/pythonjs.jpg
accent_image: 
  background: url('/assets/img/blog/pjs.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  How to automate annoying tasks with shell scripts
invert_sidebar: true
categories: programming
#tags:       [programming]
---

# sed, awk & co - master the shell



* toc
{:toc}

## sed - your search and replace function

sed stands from *stream editor* and you can imagine it as your automated search and replace function: with it you can look for patterns and replace them with other patterns. 

~~~txt
#example file: balance.txt --
- 25,13 EUR Mon Supermarket -------

+ 13,40 EUR Tue Pizza/Drinks -
- 05,00 EUR Tue Bus --

+ 40,00 EUR Wed Refund ----
~~~

Examples: 

- `sed s/,/./ <balance.txt >balance_int.txt`: read text from file1, substitute the first comma on each line with a full stop and write the output to file2
- `sed s/,/./g <balance.txt >balance_int.txt`: same as above, but with the global option `/g` sed substitute every comma with a full stop, not just the first one in each line
- `echo "15,3" | sed s/,/./`: pipe input from other commands
important: sed is searching for strings, not for words!
- `sed -i s/,/./g balance.txt`: `-i` flag makes it read and write to the same file; input/output flags not needed in this case
- `sed '/+/s/,/./g' balance.txt`: look for lines in balance.txt that contain a + and substitute , with . in these lines.
- `sed '/-/d' balance.txt`: look for lines in balance.txt that contain a - and delete these lines
- `sed -e 's/Mon/Monday/g' -e 's/Tue/Tuesday/g' -f balance.txt`: normally, sed takes first argument as expression and second input as file. In case we want to use multiple expressions and/or files, we can make this explicit with the `-e` and `-f` flags.
- `sed s/Pizza\/Drinks/Party/g`: if the search pattern itself contains a /, we can escape that with a backslash.
- `sed s#Pizza/Drinks#Part#g`: other possibility to circumvent this problem: just use other separators! sed is not very picky about which separators you use and is smart enough to understand what you are trying to do.
- `sed -n /-/p <balance.txt`: print lines from balance.txt that have a - in them. By default `sed` prints all the input it processed except for deletions. `-n` (no) suppresses this output, and the print option `/p` prints the lines that match our pattern.
- `sed -i 's/-*$//' balance.txt`: find regex pattern in each line (here dashes (`-`), an arbitrary number of them (`*`) at the end of the line (`$`)) and substitute them with nothing (`//`).
- `sed '/^$/d'`: find every empty line (nothing in between start (`^`) and end (`$`) of line) and delete it.
- `sed 's/[A-Z]/\L&/g'`: find every uppercase letter and make it lowercase. To do it the other way around, replace `[A-Z]` with `[a-z]` and `\L` with `\U`.
- `sed 10q balance.txt`: use it as replacement for `head` command. without any flags, `head balance.txt` gives you the first ten lines of a file.


It is important to use single quotes for the sed pattern instead of double quotes. If you use single quotes, sed gets exactly the pattern that you write. But when you use double quotes, the string is first passed
to the shell and interpreted by it, which can be problematic in case of special symbols and variable/command names. It can also be beneficial, but only if you know what you are doing; otherwise, stay to single quotes
(see [this thread](https://askubuntu.com/questions/1146789/single-quote-and-double-quotes-in-sed#:~:text=%40DummyHead%20Here's%20another%20approach%3A%20if,eventually%20passed%20on%20to%20sed.) for a more detailed discussion).

## awk - the allrounder

`awk` is another very powerful command line tool. Most people use it for text manipulation (similar to `sed`), but being a full scripting language, it can do a whole bunch more! Fun fact: it got its name from its three creators who wrote the tool in the AT&T Bell Labs in 1977: Alfread Aho, Peter Weinberger, and Brian Kernighan. It is especially useful if your text has some structure in it (like a tsv/csv file for example). Here some examples on what to do with it:

- `awk '{print $2}' balance.txt`: print first field/column of each line. By default, spaces separate columns in `awk` (can be customized).
- `awk '{print $0}' balance.txt`: print whole lines (equivalent to `cat`); same output if you just use `'{print}'` as command for `awk`.
- 

## getting help - man/tldr

- `man sed`: gives you the (long) manual page of sed, explaining the different options
- `tldr sed`: gives a more concise summary of the sed command, similar to a cheat sheet
- [online man page](https://www.gnu.org/software/sed/manual/sed.html) often a bit easier to read than terminal version
- great YouTube channels such as [DistroTube](https://www.youtube.com/c/DistroTube) explaining many of the tricks for shell commands; many of the example commands from this article are inspired by his videos!

~~~bash
#

~~~



## Closing thoughts

A

*[SERP]: Search Engine Results Page
