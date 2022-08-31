---
layout: post
title: Mastering Cloud Computing
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
Besides machine learning and blockchain, cloud computing is probably one of the buzzwords that have been thrown around the most in the last couple of years. But despite the hype, there are quite a lot of real advantages to using cloud computing over local on-premise resources, for example when working with highly fluctuating workloads (see [this post]() for a project I worked on at [CSIRO Sydney](https://bioinformatics.csiro.au/) that fits this description quite well).

Here I want to take a step back and look at the things that in retrospect were helpful on my learning journey of cloud computing. Along the way I will share some resources I especially liked ordered by level of prerequisite knowledge that you need, interspersed with some practical tidbits I picked up during coding and troubleshooting. Hopefully, that helps you on your learning journey, whether you are just starting or an experienced coder who wants a fresh perspective on familiar topics!

* toc
{:toc}


## Novice: What it's all about
To understand what cloud networks are about, it is very helpful to know how local networks work. I learned these concepts during my degree in lectures such as "Operating Systems" and "Computer Networks", but if you do not have these lectures in your curriculum there are lots of great online resources available. The book about the subject I enjoyed the most is [Computer Networking - A Top-Down Approach](https://gaia.cs.umass.edu/kurose_ross/index.php) by Kurose and Ross. They make the topic accessible by starting at the topmost components in the hierarchy that you are familiar with from your user experience (browser, webpages etc.) and then start to go lower and lower in the [OSI model](https://en.wikipedia.org/wiki/OSI_model), therefore allowing you to integrate the new concepts into what you already learnt beforehand. On the linked website above they provide all kinds of resources, from video lectures to excellent labs you can go through yourself to get a practical understanding of the discussed topics.

Once you have a solid understanding of computer networks, you can start dipping your feet into cloud computing, for example through introductory tutorials provided by the three major public cloud providers: [AWS](https://docs.aws.amazon.com/whitepapers/latest/aws-overview/what-is-cloud-computing.html), [Google Cloud Platform](https://cloud.google.com/learn/what-is-cloud-computing) and [Microsoft Azure](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-cloud-computing/). 

There are also great tutorials available that are agnostic to the cloud provider: for that, you can check out courses on [ACloudGuru](https://acloudguru.com/course/introduction-to-cloud-computing) or [Udemy]( https://www.udemy.com/course/introduction-to-cloud-computing-on-amazon-aws-for-beginners/). But honestly, for an introduction to the topic I would not bother with paid courses, but just use what is freely available on YouTube such as [this](https://www.youtube.com/watch?v=RWgW-CgdIk0) or [this](https://www.youtube.com/watch?v=_a6us8kaq0g) video.

## Intermediate: Choose your platform
Once you understand what cloud computing is and you want to get practical experience, I would recommend choosing one of the major public cloud providers mentioned above to dig deeper into how to use their platform and get your hands dirty with some toy projects. As in any domain, doing instead of just reading will improve your skills, so make sure to choose a project to directly put your learnings into practice!

The services the three major players provide often just differ in their names, so make sure to not only understand the specific terms introduced by the platform you are using but to connect it to the general networking concepts you know about and to see what other provides have in store for the same purpose.

As an example, let's consider database solutions. AWS has an overwhelming amount of these, ranging from RDS, Aurora, Aurora Serverless, and Amazon Redshift to DynamoDB, DocumentDB or Amazon Neptune. Instead of learning all these names and their specifications, remember the two general categories of databases out there: relational (SQL) and non-relational (NoSQL) databases. 

Once you have this general framework and what this means for applying one or the other option in your mind, you can just put the different services in the right mental drawers: RDS, Aurora and Redshift are relational databases whereas DynamoDB, DocumentDB and Neptune are non-relational database services. 

This mental framework not only makes it easier to understand what to use each service for but also how to compare the services to other providers: Google Cloud Platform for example offers CloudSQL and CloudSpanner as managed relational database services and Cloud Bigtable as well as Cloud Firestore as non-relational databases.

There are tons of resources to get familiar with a particular cloud platform. The providers themselves have tutorials on their websites, but I especially like the [ACloudGuru](https://acloudguru.com/) platform for two reasons.

First, they offer both general knowledge courses and specialised certification courses, striking a balance between the aforementioned background skills and provider-specific knowledge. 

Second, they incorporate hands-on labs in their courses which allow you to play in a safe sandbox environment with real provider interfaces. You can also do that with a free account on AWS, but watch out: if you leave your resources running and do not terminate them properly, you might reach the free tier limit and get a hefty AWS bill! There are [horror stories](https://twitter.com/flaviocopes/status/1542148015808544769) on Twitter&Co about people getting crazy bills overnight. No matter how much of this is true: A sandbox environment gives you a safe space to explore what you can do with cloud computing.

Courses I liked were AWS Certified Developer Associate and the Linux Networking & Troubleshooting course. Because one thing is for sure: no matter whether you do cloud computing, use a local cluster or just work on your desktop, troubleshooting will be a major part of your work; worth taking the time to get good at it.

## Master: Specialise and find your niche
Once you mastered one (or even multiple) of the major cloud platforms, it is usually helpful to focus your learning on what your end goal is. Do you want to develop ML applications? Choose the AWS ML speciality. Are you into databases? Choose the AWS databases speciality. 
The options are endless, so it is important to keep a focus and not get distracted by the flood of information available. 

Here again, it helps to have a concrete project to work on: [be a better dev](https://www.youtube.com/c/BeABetterDev) is a great YouTube channel which has resources to [choose a cloud project](https://www.youtube.com/watch?v=06VgLTqNvU8) as well as explainer videos for many of the concepts needed for these projects. 

As always, [Free Code Camp](https://www.freecodecamp.org/news/tag/cloud-computing/) is a great resource for everything related to coding; they have some great projects on their website you can start implementing, such as [this one](https://www.freecodecamp.org/news/learn-terraform-and-aws-by-building-a-dev-environment/) in which you will use Terraform and AWS to build your own Dev environment.

## Useful tools 

Talking about Terraform: there are a lot of concepts not specific to cloud computing that will nevertheless be very helpful for your journey as a cloud developer. I mentioned networking & troubleshooting above, but others include Docker, Infrastructure as Code (such as Terraform) or general advanced coding knowledge. In the end, every cloud developer is just a developer, so you better be familiar with these developer tools as well!


## Closing thoughts

I know I mentioned this quite a lot in this post, but it is worth reiterating: choose a project that excites you while learning about cloud computing! You will be more motivated and better at the actual task of being a developer: developing. 

For example, I started with some toy projects on ACloudGuru but quickly moved to a task during my internship where I reimplemented a [software for detecting DNA integration](https://bioinformatics.csiro.au/blog/detecting-foreign-dna-with-insider/) as a serverless service. In the course of it, I learned a lot about all different concepts and even wrote [my own Terraform module](https://registry.terraform.io/modules/kierandidi/emrserverless/aws/1.0.0) in the process! So believe me, doing is the best way to learn.

Hope this post was helpful for you, good luck on your learning journey and happy coding!



*[SERP]: Search Engine Results Page