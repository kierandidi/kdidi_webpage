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

You probably know the pain: You read a cool paper on arxiv or see a cool post on Twitter and want to play around with the code. So you clone the GitHub repo linked by the authors, install the required dependencies and you are ready to go, right? Unfortunately it often isn't that easy in reality: Sometimes the authors do not specify which versions of the packages you need; other times their environment file is OS-specific and you cannot use it on your system; and sometimes the libraries needed are even not available for your system. So, all hope lost?

Luckily not: with [containers](https://en.wikipedia.org/wiki/OS-level_virtualization), people can provide a snapshot of their system that other people can easily use to run the same code, without the hassle of installing the right package version and so on! [Docker](https://www.docker.com/) is probably the most common tool for creating and using containers, so in this tool I will give you an overview over how Docker works and a cheat sheet of the most common commands you may want to use for your containers.

* toc
{:toc}

## Installation

[Docker extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)

## Management Commands
- `docker -h`: print out help page 

All docker commands are grouped under the umbrella of so-called *management commands*. These commands describe the building blocks on which Docker is built and give us a better overview of the structure of the Docker application itself. For example, there was `docker ps`, but what is that actually showing us? Containers? Images? Dockerfiles? The "newer" version of it is now `docker container ls`; here you can clearly see that we list the containers due to the management command `docker`.
There are [twelve management commands in total](https://www.couchbase.com/blog/docker-1-13-management-commands/), but we will focus on a subset of these to start with. 

Short side note: most of the objects in docker like containers, networks and volumes have both a name and an ID. In most commands you find in this post you can use the two interchangably, with the exception of creating a resource: here you can specify its name, but the ID is set by Docker itself.

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

#### Executing commands on container
- `docker container attach <container id>`: attach shell to current container. Watch out: once you detach, Docker will stop the container! Problem: our output is not yet directed to the shell, but to the logs.
- `docker container logs <container id>`: get log data from container (by default standard output of container)
- `docker container stats <container id>`: gives you information about resource usage etc. Exit from it via `Control` + `C`
- `docker container exec -it <container-id> /bin/bash`: run a command in a specified container that is currently running. `-i` flag makes the process interactive and `-t` allocates a pseudo TTY to the container. In this example, we get access to the bash shell of the container and can run commands from there. With `exit` you can leave the container and come back to your shell, but the container will still be running.


#### Cleanup
- `docker container rm <container-id>: remove a docker container. If you want to remove a running container, you have to force it via the `-f` flag.
- `docker container prune`: removes all stopped containers (has to be confirmed in terminal). With `-f` flag, you skip the confirmation step.

#### Container Ports
- `docker container run -d --expose 2000 nginx`: exposes port, makes it available to be mapped (via `-p` or `-P` flags), but is not published yet!
- `docker container run -d --expose 2000 -p 80:2000 nginx`: exposes port and publishes it (maps an exposed container's port to a host's port). Here, we map the container port 2000 to the host port 80. Still `curl localhost:2000` gives connection refused since we do not have a process listening at that port on the container! But if we change the `-p` flag option to e.g. `8000:80`, `curl localhost:8000` gives us the expected response from `nginx` since on port 80 of the container the nginx port is listening.
- `docker container run -d  -p 80:1999/tcp -p 80:1999/udp nginx`: you can specifiy multiple port mappings of exposed ports with specific protocols.
- `docker container run -d -P nginx`: the `-P` flag publishes all exposed ports on the container to random ports on the host.
- `docker container port <container-id>`: list all port mappings for a container

#### Executing container commands
 
 There are three ways to execute a command on a docker container: via the Dockerfile (with the directive `CMD`), during a docker run via `docker container run` or using the `exec` command (which will run in the default directory of the container and only if the primary process of the container is still running). 

When we start a container, the Docker container either behaves like a task (stops quite quickly) or like a process if the primary process of the container is a long-running one (in this case the container keeps on in running mode, like our nginx container).

- `docker container run -d nginx`: starts container in background due to `-d` flag
- `docker container run -it nginx`: starts container with interactive TTY, but we cannot execute anything since we did not specify any command that should be launched in interactive mode! Exit via `exit` command.
- `docker container run -it nginx /bin/bash`: starts container interactively and starts up its bash shell, which we can use to execute programs on the container.
- `docker container exec -it <container-id> ls`: we do not have to enter the Docker container to execute commands, but can do it via the `exec` command that just sends our command to the container and executes it for us there.

#### Container logging

- `docker container logs <container-id>`: show logging information of a running container. Good for troubleshooting if for example the Docker container was not able to start due to an error.
- `docker service logs <service>`: if we use services (as shown later) via Docker Swarm, multiple containers participate in a single service. with this command, we get the logs from all containers that are involved in the specified service.

#### Docker Networking

Three major components: Container Netowrk Model (CNM), libnetwork and network drivers (bridge, host, overlay, macvlan, none and other network plugins available via Docker Hub).

Three building blocks that the CNM defines: 
- sandboxes: isolates network stack
- endpoints: virtual network interfaces to connect sandbox to network
- networks: software implementations of bridge

- `ifconfig`: shows you the configuration of your network interface and allows you to change it.
- `docker network ls`: lists all networks present. By default there are three: bridge, hoste and none. Do not deinstall these or you might have to reinstall Docker itself!
- `docker network inspect <network-id>`: shows detailed information for a specified network in JSON format.
- `docker network create <network-name>`: create a new network with specified name (by default driver will be `bridge`). We can add the `--internal` flag to tell Docker not to connect this network to any of our available interfaces. If you do that, you will not be able to `curl` or `ping` the container connected to this network by specifying the port, but by specifying its private IP since you and the container are in the same private subnet.
- `docker network create --subnet <subnet CIDR range> --gateway <gateway IP> <network-name>`: create a network with specified subnet and gateway. For the subnet we need to use private and not public IP ranges; so `10.`, `172.` and `192.`.
- `docker network create --subnet <subnet CIDR range> --gateway <gateway IP> <network-name> --ip-range <intended CIDR range> --driver <driver type> --label <label for network>`: same as above, but specifies a label for the network as well as a driver (for example `bridge`). we also tell the network the IP range we intend to use via the `--ip-range` flag; this of course has to be smaller than the IP range supplied to `--subnet` since we cannot use more IP addresses than present in our subnet.
- `docker container run -it <image> --network <network-id> <CMD>`: creates a container connected to a specified network.
- `docker container run -it <image> --network <network-id> --ip <IP> <CMD>`: same as above, but in addition container receives a specific IP.

- `docker network rm <network-name>`: delete a specified network
- `docker network prune`: deletes all unused networks (risky!)
- `docker network connect <network-id> <container-id>`: connects specified network with specified container (which is by default already connected to `bridge`).
- `docker network disconnect <network-id> <container-id>`: disconnects specified network from specified container.

## Storage and Volumes with Docker Containers

There are two different kinds of storage associated with Docker containers: non-persistent and persistent storage.

1. **Non-persistent data storage** is available to every container. It only exists during the lifecycle of a container, meaning that all the data stored there is lost when the container is deleted. Therefore, data on there should be data that is tied to the lifetime of the container as well and should not need to survive the container itself; application code is a good example for that. Other names for non-persistent data storage are *local storage*, *graph driver storage* or *snapshot storage*. The storage location is `/var/lib/docker/<storage-driver>` under Linux and `c:\ProgramData\Docker\windowsfilter\` under Windows. The storage driver depends on the operating system, but both Red Hat Enterprise Linux and Ubuntu use `overlay2` (with Ubuntu having the option to use `aufs` as well).

2. **Persistent data storage** is referred to as *Volumes*. Volumes are not connected to the lifecycle of a container, meaning that the data stored there is still present if the container is deleted. The default process to using volumes is to first create a volume, then create a container and finally mounting the volume to a directory in the container. The data can then be written to this directory and is safely stored on the volume even when the container is deleted. Since volumes are first-class citizens in Docker, they can be accessed in the same way as containers or networks: via management commands (often also called subcommands). The volume location is `/var/lib/docker/volumes/` under Linux and `c:\ProgramData\Docker\volumes` under Windows. By default volumes use the local driver from the Docker server, but third-party drivers utilising different kinds of storage (object storage, block storage or file storage) can also be used instead.

- `docker volume -h`
- `docker volume ls`
- `docker volume create <volume-name>`
- `docker volume create <volume-id>`
- `docker volume inspect <volume-id>`
- `docker volume rm <volume-id>`
- `docker volume prune`: 

Beside volumes, we can also use a more specialised construct for persistent data storage called **bind mounts**. Bind mounts serve to mount a file or directory that already exists on the host into a container, whereas volumes create a new directory on the container that is located within Docker's storage directory; Docker is managing this directory, whereas the host machine manages the bind mount directory.

- `docker container run -d --mount type=bind,source="$(pwd)"/<directory-name>,target=/app <image>`: creates a container from a specified image in background mode and creates a bind mount on it with the folder `<directory-name>` from our current working directory on the host machine. This directory is going to be saved on the docker container in the folder we specify as `target`, in this case `/app`. The directory we want to mount has to exist before executing this command, otherwise we get an error. Once mounted the directory exists on both the Docker container and the host machine. Bind mounts are often useful for a short one file transfer.
- `docker container run -d --volume "$(pwd)"/target2:/app <image>`: same as above, but instead of the bind mount we create a volume at the specified location in the container. Since the volume will be create anew in the container we do not have to have a directory with this name existent on our host machine. On the Docker container we will have the volume called `/app`, wheras we will have it as `/target2` saved in our curent working directory on the host machine. Instead of `--volume` we can also use the shorter volume flag `-v`. We could have used the more verbose option with the `mount` flag case by replacing `type=bind` with `type=volume`, but I just wanted to show you the more concise location with the colon separating source and target. We could also appen the option `,readonly` after the target while using the mount flag; this prevents us from making changes to this volume while on the container.

Volumes are often preferred over bind mounts when you want to use this storage space regularly and not just once, since they are easier to migrate to other locations and easier to back up than bind mounts.

## Dockerfile

We talked a lot about converting images into containers and different ways to do this via the `run` command, but how do create these images in the first place? That is where the *Dockerfile* comes into play: it contains the instructions on how to build a Docker image. 

You can imagine a Docker image as a pile of images stacked up onto each other that are read-only; so you can build and run a container from it, but not change the image itself. 

<p align="center">
  <img src="/assets/img/blog/docker/book_pile.png" width="50%" height="50%"/>
</p>

*Ahe Dockerimage is like a pile of objects that were built using instructions from the Dockerfile (the objects are represented by bricks in this case).*

The Dockerfile is responsible for changes: each layer in the Docker image is built via an instruction from the Docker file. Thereby, the image is created iteratively by building layer upon layer, each layer represententing the changes compared to the previous layer. 

<p align="center">
  <img src="/assets/img/blog/docker/brick-pile.png" width="50%" height="50%"/>
</p>

*A Dockerfile is like a big pile of instructions (for example how to produce bricks). From these instructions, a Docker image can be built.*

Every Dockerfile has a structure comparable to the one below:

~~~bash
# file: dockerfile.doc

~~~


## Closing thoughts

A

*[SERP]: Search Engine Results Page
