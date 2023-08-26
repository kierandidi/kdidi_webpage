---
layout: post
title: Debugging on a GPU node
image: /assets/img/blog/ssh_gpu/gpucluster.jpg
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  How to log into and debug on a GPU node
invert_sidebar: true
categories: tools
---

# Debugging on a GPU node

In a [previous post](https://kdidi.netlify.app/blog/tools/2022-10-22-ssh-in-vscode/) I discussed some ways to make the experience of working on a remote machine via SSH more pleasant in VSCode by using an extension for VS Code called [Remote SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh). With this extension, you can not only interact with the remote machine via the integrated terminal, but also have access to the filesystem via drag-and-drop in the sidebar as well as other perks.

One problem persists though: If the machine you connect to is a compute cluster such as typically present at universities, chances are the node you are connected to is a login node, not a compute node. This is not a problem if you simply want to use the workspace features of the extension I mentioned above; however, if you want to use VS Code to actually run (and, more importantly, debug) code, you do not want to do this on the login node but on a dedicated (CPU or GPU) compute node. In this post, I will describe the difficulties one can encounter with this setup and how to resolve them.

As a short recap from the last post, the main three steps to set up SSH via VS Code are:
1. Set up the ssh config file
2. Set up your ssh keypair
3. Configure the `Remote SSH` extension in VS Code

Now, 

#### The problem: Login versus compute node

To avoid typing in the IP or long hostname of your remote machine every time you want to connect, we will create an ssh config file to contain that information. 
If you do not already have a `.ssh` directory on your local machine, create one via `mkdir ~/.ssh`.
Since no one else should use the information in this config file beside you, it is recommended to give the directory owner-only permission via `chmod 700 ~/.ssh`.

We can then set up the config file by typing `nano ~/.ssh/config` which will open the nano editor. In this file you can specifiy as many remotes as you want with the corresponding information on how to connect with them. An example file could look like this:

~~~bash
# file: "/.ssh/config"
# Host information
Host remote123
  User kieran
  Hostname 130.183.194.13
  IdentityFile ~/.ssh/remote123_rsa
~~~

For the config file you need:
- abbreviation for the host (up to you, here remote123)
- the username with which you logon to the remote machine
- the IP address of the remote machine
- the identitify file with the public key (we will deal with this later)

#### The solution: repeat the process

In order to avoid retyping our password every time we login, we can use what is called public-private encryption (more on that for example [here](https://missing.csail.mit.edu/2020/security/)). 

To do this, we first create a keypair on our local machine by typing `ssh-keygen` into our terminal. As a filename for the pair I will use remote123_rsa (as I specified beforehand in the config file), and when it prompts me to enter a passphrase I just hit `Return` to avoid having a passphrase, but this is up to you.

Now you should have two files on your local machine: the public key `remote123_rsa.pub` which can be seen as a lock and the private key `remote123_rsa` which can be seen as a key. In this kind of encryption, you will give the lock to the remote machine, but do not share the key with anyone. Every time you log into the remote machine, you will basically show it that you can unlock the lock you gave it earlier with your private key, proving that you are allowed to access your account.

For that to work, we have to share the public key with the remote machine. For that, copy your public key to your clipboard via `cat remote123_rsa.pub | pbcopy`.

Now, log into your remote machine and execute the following series of commands:

- `mkdir ~/.ssh` and `chmod 700 ~/.ssh` in case there is no `.ssh` directory yet.
- `touch ~/.ssh/authorized_keys`
- `chmod 600 ~/.ssh/authorized_keys`
- `nano ~/.ssh/authorized_keys`

With this, you should now again be in a nano editor opened. Paste your public key in there via `Cmd + V`, save the file and voilà! You should now be able to connect to your remote machine via just running `ssh remote123`.

#### Problem 2: 2 factor authentication

This last step is actually trivial: You can just open the remote-ssh extension, point the extension towards your ssh config file on your local machine and you're done! So now you can just open that extension in VS Code, right-click on the host you want to connect and choose `Connect`; it will open the SSH tunnel and allow you to manipulate files and open terminals in VS Code but on the remote machine.

<p align="center">
  <img src="/assets/img/blog/mac_setup/remote-connect.png"/>
</p>

Once you have this working, VS Code unlocks its powers: you can connect to your remote machine in a few clicks and directly have an integrated terminal at your disposal. Even better, you can just copy files from and to your remote machine just by using drag-and-drop! For me this was a real game changer.

### Solution 2: 

## Closing thoughts

Working with remote machines can be daunting if you just heard of how to do it and feel overwhelmed by public/private keys, ssh etc. But once you set it up properly, it actually feels no different to working locally! You have your favourite editor still at hand, can easily switch between local and remote files and start enjoying the power of remote machines. Happy coding!



*[SERP]: Search Engine Results Page