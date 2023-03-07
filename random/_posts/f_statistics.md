---
layout: post
title: The F is everywhere
image: /assets/img/blog/jj-ying.jpg
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  F statistics vs the F distribution vs the F test vs the F ratio
invert_sidebar: true
categories: [random]
tags:       [events]
---

# The F is everywhere

In a recent lecture on theoretical population genetics, I heard about the F statistic in the context of inbreeding. I was quite confused since my association with F statistics was with variance analysis and not with the inbreeding of European royalty. So I dug a little deeper and (after some initial confusion) cleared up the nomenclature a bit.

## F statistics (F<sub>IS</sub>, F<sub>ST</sub>, F<sub>IT</sub>; Wright's F)

The F statistics are used in population genetics to measure the degree of inbreeding in a population. If you assume random mating between individuals (i.e. no inbreeding) and a few other things, then we could describe our allele frequencies with the [Hardy Weinberg equilibrium]() formula:

$$
p^2 + 2pq + q^2 = 1
$$

where $p$ is the frequency of the dominant allele and $q$ is the frequency of the recessive allele. So in this case $$2pq$$ is the frequency of heterozygotes. That would be all nice and simple, but biology is unfortunately not nice and simple but pretty damn complicated. 

First of all (as you might have guessed already) people do not mate randomly; there are effects like local subpopulation structure that we need to take into account to model reality. But even beyond that, inbreeding (i.e. ) is pretty common in nature. Many plants and animals are self-fertile and will mate with their own offspring. This is not a problem in itself, but it does lead to a higher frequency of homozygotes and a lower frequency of heterozygotes. So if we want to model reality, we need to take this into account.

The F statistics are used to measure this: the degree of inbreeding in a population. We can start with $$F$$ for the inbreeding coefficient, which is defined as the ratio of the observed heterozygosity to the expected heterozygosity:

$$
\begin{aligned}
F = \frac{H_e - H_o}{H_e}
= \frac{2pq - H_o}{2pq}
\end{aligned}
$$

where $$H_o$$ is the observed heterozygosity and $$H_e$$ is the expected heterozygosity. We can interpret $$F$$ as the probability of a random mating between two individuals in the population to produce a homozygote. So $$F=0$$ means that there is no inbreeding and $$F=1$$ means that all individuals are homozygotes. This total $$F$$ is also known as $$F_{IT}$$, to indicate that it is the inbreeding coefficient of an individual (I) relative to the total (T) population.

But as I mentioned above, population substructures have an effect on mating and inbreeding as well. 
So we assume a population structure with two levels: one going from the total population $$T$$ to subpopulations $$S$$ and one going from subpopulations $$S$$ to individuals $$I$$.

That means that we can partion our $$F_IT$$ into two parts: $$F_{IS}$$ and $$F_{ST}$$. $$F_{IS}$$ is the inbreeding coefficient of an individual (I) relative to the subpopulation (S) it is in. $$F_{ST}$$ is the inbreeding coefficient of a subpopulation (S) relative to the total population (T).

$$
\begin{aligned}
F_{IT} = F_{IS} + F_{ST}
\end{aligned}
$$

So we can define $$F_{ST}$$ as the fraction of genetic variance (i.e. heterozygosity) that is due to differences between subpopulations:

$$
\begin{aligned}
F_{ST} = \frac{H_{T} - H_{S}}{H_{T}}
\end{aligned}
$$

where $$H_{T}$$ is the total heterozygosity and $$H_{S}$$ is the heterozygosity within subpopulations. So $$F_{ST}$$ is the fraction of heterozygosity that is due to differences between subpopulations. We define $$F_{IS}$$ as the fraction of genetic variance (i.e. heterozygosity) that is due to differences between individuals within a subpopulation:

$$
\begin{aligned}
F_{IS} = \frac{H_{S} - H_{I}}{H_{S}}
\end{aligned}
$$

where $$H_{I}$$ is the heterozygosity within individuals. 

Together these statistics give us a good idea of the degree of inbreeding in a population and how it is distributed across the population structure.

## F values (F<sub>2</sub>, F<sub>3</sub>, F<sub>4</sub>; Patterson's F)



## F distribution

## F test/statistic (Fisher's F)

There is another F statistic that is used in variance analysis. This is the F statistic that is used in the F test. The F test is used to test the significance of the overall effect of a factor in a variance analysis. Whereas the t test is used to test the significance of the effect of a single factor, the F test is used to test the joint significance of the effect of multiple factors.

As with most test statistics, the F test is based on a test statistic that is distributed according to a certain distribution. In this case, the F test is based on the F statistic, which is distributed according to the F distribution. To test the significance of the overall effect of a factor, we calculate the F statistic and compare it to the F distribution. If the F statistic is larger than some critical value, we can reject the null hypothesis that the overall effect of the factor is not significant.



## F ratio


## Credits

<span>Photo by <a href="https://unsplash.com/@jjying?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">JJ Ying</a> on <a href="https://unsplash.com/?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

*[SERP]: Search Engine Results Page
