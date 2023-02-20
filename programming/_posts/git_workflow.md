---
layout: post
title: Git - 
image: /assets/img/blog/cloud_bits.png
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Resources for learning cloud computing, from novice to master
invert_sidebar: false
categories: programming
#tags:       [programming]
---

# Mastering Cloud Computing


* toc
{:toc}


# Introduction

## Pull changes from remote

In order to pull changes from a remote repository, you need to add the remote repository as a remote to your local repository. You can do this by running the following command:

    git remote add <remote_name> <remote_url>

To pull changes from the remote repository, you need to run the following command:

    git pull <remote_name> <branch_name>

This often results in a merge conflict, which you need to resolve before you can push the changes to the remote repository. You can resolve the merge conflict by running the following command:

    git mergetool

Sometimes you might want to pull changes from a remote repository without merging them into your local repository. You can do this by running the following command:

    git fetch <remote_name> <branch_name>

Other times you might want to pull changes from a remote repository without merging them into your local repository and without updating your local repository's HEAD. You can do this by running the following command:

    git fetch --no-head <remote_name> <branch_name>

Other times again you might just want to pull changes from a remote repository and merge them into your local repository and just go on with your life, overwriting your changes with the changes from the remote repository. A relatively clean way to do this is via a two-step process:

1. Commit your current changes so they are not lost in case you want to revert them later:
`git add *`
`git commit -m "Committing my local changes"`

2. Fetch the changes from the remote repository and overwrite them if there is a conflict:
`git fetch origin <branch-name>`
`git merge -s recursive -X theirs origin/<branch-name>`

X here is a flag that tells git to use the changes from the remote repository in case of a conflict (theirs) instead of the changes from your local repository (ours).






## Closing thoughts





*[SERP]: Search Engine Results Page