---
layout: post
title: How to Docker
image: /assets/img/blog/docker/container.jpg
accent_image: 
  background: url('/assets/img/blog/pjs.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Use Docker to make your applications truly reproducible
invert_sidebar: true
categories: programming
#tags:       [programming]
---

# How to Docker

## Management Commands
- `docker -h`: print out help page 

All docker commands are grouped under the umbrella of so-called *management commands*. These commands describe the building blocks on which Docker is built and give us a better overview of the structure of the Docker application itself. For example, there was `docker ps`, but what is that actually showing us? Containers? Images? Dockerfiles? The "newer" version of it is now `docker container ls`; here you can clearly see that we list the containers due to the management command `docker`.
There are [twelve management commands in total](https://www.couchbase.com/blog/docker-1-13-management-commands/), but we will focus on a subset of these to start with.

### Images 

- `docker image -h`: get all commands associated with `image` management command
- `docker image ls`: show all current images, including ID etc
- `docker image pull hello-world`: pull the image `hello-world` from the Docker Hub
- `docker image inspect <instance id>`: show information about the image with the specified image ID in the terminal as JSON format

### Containers

#### Basics
- `docker container -h`:
- `docker container ls`: show all running containers. With the flag `-a` it shows all containers, including stopped ones.

#### Running and Stopping
- `docker container run hello-world`: runs the `hello-world` container from the pulled image. The container will execute the `sh` command by default and then exit and stop. If you want to keep your container list clean and remove containers automatically once they stop, you can append the `--rm` flag to the run command.
- `docker container run -P -d hello-world`: the `run command has several config options: 
  - `-P` takes all available port numbers and maps our docker container to a random one from this list. 
  - `-d` puts detaches us from the container (putting it in the background) so that we can continue running processes in the terminal (not important for this example; if you want to have a longer running one try `nginx` for example). You can check if it worked by using `curl localhost:<assigned-port>`. 
  - `--name`: optionally you can give your container a custom name 
- `docker container inspect <container id>`: inspect specific container (specify either container ID or name)
- `docker container top <container id>`: list processes running on a specific container
- `docker container stop <container id>`: stops a running container, similar to shutting down a computer
- `docker container start <container id>`: starts a stopped container, similar to booting up a computer
- `docker container pause <container id>`: pauses a running container, meaning all running processes are paused
- `docker container unpause <container id>`: unpauses a paused container, meaning all paused processes resume

#### Executing commands
- `docker container attach <container id>`: attach shell to current container. Watch out: once you detach, Docker will stop the container! Problem: our output is not yet directed to the shell, but to the logs.
- `docker container logs <container id>`: get log data from container (by default standard output of container)
- `docker container stats <container id>`: gives you information about resource usage etc. Exit from it via `Control` + `C`
- `docker container exec -it <container-id> /bin/bash`: run a command in a specified container that is currently running. `-i` flag makes the process interactive and `-t` allocates a pseudo TTY to the container. In this example, we get access to the bash shell of the container and can run commands from there. With `exit` you can leave the container and come back to your shell, but the container will still be running.

#### Cleanup
- `docker container rm <container-id>: remove a docker container. If you want to remove a running container, you have to force it via the `-f` flag.
- `docker container prune`: removes all stopped containers (has to be confirmed in terminal). With `-f` flag, you skip the confirmation step.

#### Container Ports
- 

#### Making containers

- 

* toc
{:toc}

## Installation

## Basic Usage

## Advanced 






## Closing thoughts

A

*[SERP]: Search Engine Results Page
