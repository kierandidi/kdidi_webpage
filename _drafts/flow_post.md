---
layout: post
title: (GER) Normalizing Flows Teil 1 - Daten und Determinanten
image: /assets/img/blog/flows_chart.png
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
Was Normalizing Flows sind und wie man sie sich vorstellen kann
invert_sidebar: true
---

# (GER) Normalizing Flows Teil 1 - Daten und Determinanten

(Die deutsche Version beginn unten!)

This post is a rather unusual one since it is in German. I have always been involved in making content available in other languages to allow more people to enjoy it, such as when I did translations for Khan Academy. The following post is a translation of an excellent [post on normalizing flows](https://blog.evjang.com/2018/01/nf1.html) by Eric Jang, who fortunately shares my passion for making content available in different languages. So for all German-speaking folks among you - Lasst uns loslegen!


* toc
{:toc}

## Einführung: Was und für wen?

Wenn du an maschinellem Lernen, an generativer Modellierung, Bayesian Deep Learning oder Deep Reinforcement Learning arbeitest, sind "Normalizing Flows" eine praktische Technik, die du in deinem algorithmischen Werkzeugkasten haben solltest.

Normalizing flows transformieren einfache Dichteverteilungen (wie die Normalverteilung) in komplexe Verteilungen, die für generative Modelle, RL und Variational Inference verwendet werden können. TensorFlow hat ein paar nützliche Funktionen, die es einfach machen, Flows zu erstellen und zu trainieren, um sie an reale Daten anzupassen.

Diese Serie besteht aus zwei Teilen:
- Teil 1: Daten und Determinanten. In diesem Beitrag erkläre ich, wie invertierbare Transformationen von Dichteverteilungen verwendet werden können, um komplexere Dichteverteilungen zu implementieren, und wie diese Transformationen miteinander verkettet werden können, um einen "Normalizing Flow" zu bilden.

- Teil 2: Moderne Normalizing Flows: In einem Folgebeitrag gebe ich einen Überblick über die neuesten Techniken, die von Forschern entwickelt wurden, um Normalizing Flows zu erlernen, und erkläre, wie eine Reihe von modernen generativen Modellierungstechniken - autoregressive Modelle, MAF, IAF, NICE, Real-NVP, Parallel-Wavenet - alle miteinander in Beziehung stehen.
Diese Reihe ist für ein Publikum mit einem rudimentären Verständnis von linearer Algebra, Wahrscheinlichkeit, neuronalen Netzen und TensorFlow geschrieben. Kenntnisse über die jüngsten Fortschritte im Bereich Deep Learning und generative Modelle sind hilfreich, um die Motivationen und den Kontext, der diesen Techniken zugrunde liegt, zu verstehen, aber sie sind nicht notwendig.


## Hintergrund 

Statistische Algorithmen für maschinelles Lernen versuchen, die Struktur von Daten zu erlernen, indem sie eine parametrische Verteilung $$ p(x;θ) $$ an sie anpassen. Wenn wir einen Datensatz mit einer Verteilung darstellen können, können wir:

1. Neue Daten "kostenlos" generieren, indem aus der gelernten Verteilung in silico Stichproben gezogen werden ("sampling"); es ist nicht notwendig, den eigentlichen generativen Prozess für die Daten durchzuführen. Dies ist ein nützliches Werkzeug, wenn die Daten teuer zu generieren sind, z. B. bei einem realen Experiment, dessen Durchführung viel Zeit in Anspruch nimmt [^1]. Sampling wird auch verwendet, um Schätzer für hochdimensionale Integrale über Räume zu konstruieren.

2. Bewertung der Wahrscheinlichkeit der zum Testzeitpunkt beobachteten Daten (dies kann für Rejection Sampling verwendet werden oder um zu bewerten, wie gut unser Modell ist).

3. Ermittlung der bedingten Beziehung zwischen Variablen. Das Erlernen der Verteilung $$ p(x_2|x_1) $$ ermöglicht es uns zum Beispiel, diskriminierende (im Gegensatz zu generativen) Klassifizierungs- oder Regressionsmodelle zu erstellen.

4. Bewertung unseres Algorithmus anhand von Komplexitätsmaßen wie Entropie, gegenseitige Information und Momenten der Verteilung.

Wir sind ziemlich gut im Sampling (1) geworden, wie die jüngsten Arbeiten an generativen Modellen für Bilder und Audio zeigen. Diese Art von generativen Modellen wird bereits in echten kommerziellen Anwendungen und Google-Produkten eingesetzt. 

Allerdings widmet die Forschungsgemeinschaft der unbedingten und bedingten Wahrscheinlichkeitsschätzung (2, 3) und dem Model-Scoring (4) derzeit weniger Aufmerksamkeit. Wir wissen zum Beispiel nicht, wie man den Träger eines GAN-Decoders berechnet (wie viel des Ausgaberaums vom Modell mit einer Wahrscheinlichkeit ungleich Null belegt wurde), wir wissen nicht, wie man die Dichte eines Bildes in Bezug auf eine DRAW-Verteilung oder sogar einen VAE berechnet, und wir wissen nicht, wie man verschiedene Metriken (KL, Earth-Mover-Distanz) für beliebige Verteilungen analytisch berechnet, selbst wenn wir ihre analytischen Dichten kennen.

Es reicht nicht aus, wahrscheinliche Stichproben zu erzeugen: Wir wollen auch die Frage beantworten: "Wie wahrscheinlich sind die Daten?" [^2], flexible bedingte Dichten (z. B. für die Stichprobenbildung und die Bewertung von Divergenzen multimodaler Policies in RL) und die Möglichkeit, umfangreiche Familien von A-priori Wahrscheinlichkeiten ("priors") und "posteriors" in Variational Inference zu wählen. 

Lasst uns für einen Moment den netten Nachbarn von nebenan anschauen: Die Normalverteilung. Sie ist der Klassiker unter den Verteilungen: Wir können leicht Stichproben aus ihr ziehen, wir kennen ihre analytische Dichte und KL-Divergenz zu anderen Normalverteilungen, der zentrale Grenzwertsatz gibt uns die Gewissheit, dass wir sie auf so gut wie alle Daten anwenden können, und wir können sogar mit dem Trick der Reparametrisierung durch ihre Stichproben Backpropagation durchführen (siehe VAEs). Diese netten Eigenschaften der Normalverteilung macht sie zu einer sehr beliebten Wahl für viele generative Modellierungs- und RL-Algorithmen.

Leider ist die Normalverteilung bei vielen realen Problemen, die uns interessieren, einfach nicht geeignet. Beim Reinforcement Learning - insbesondere bei kontinuierlichen Steuerungsaufgaben wie in der Robotik - werden Strategien oft als multivariate Normalverteilungen mit diagonalen Kovarianzmatrizen modelliert. 

Per Definition können uni-modale Normalverteilungen bei Aufgaben, die eine Stichprobenziehung aus einer multimodalen Verteilung erfordern, nicht gut abschneiden. Ein klassisches Beispiel dafür, wo uni-modale Strategien versagen, ist ein Agent, der versucht, über einen See zu seinem Haus zu gelangen. Er kann nach Hause gelangen, indem er den See im Uhrzeigersinn (links) oder gegen den Uhrzeigersinn (rechts) umgeht, aber eine Gauß'sche Strategie ist nicht in der Lage, zwei Modi darzustellen. Stattdessen werden Aktionen aus einer Gaußschen Kurve ausgewählt, deren Mittelwert eine lineare Kombination der beiden Modi ist, was dazu führt, dass der Agent geradewegs in das eisige Wasser läuft. Traurig!

Das obige Beispiel veranschaulicht, wie die Normalverteilung zu vereinfachend sein kann. Zusätzlich zu den schlechten Symmetrieannahmen konzentriert sich die Dichte der Gauß-Verteilung in hohen Dimensionen größtenteils auf die Ränder und ist nicht robust gegenüber seltenen Ereignissen. Können wir eine bessere Verteilung mit den folgenden Eigenschaften finden?

1. Komplex genug, um reichhaltige, multimodale Datenverteilungen wie Bilder und Wertfunktionen in RL-Umgebungen zu modellieren?
2. ... unter Beibehaltung der netten Eigenschaften einer Normalverteilung: Stichprobenziehung, Dichteauswertung und mit reparametrisierbaren Stichproben?

Die Antwort ist ja! Hier sind ein paar Möglichkeiten, dies zu tun:
- Verwendung eines Mischmodelles (siehe GMMs), um eine multimodale Policy zu repräsentieren, wobei eine kategorische Variable die "Option" und eine Mischung die Subpolicy repräsentiert. Auf diese Weise erhält man Stichproben, die einfach zu ziehen und auszuwerten sind. 
Es gibt aber ein Problem: Die Stichproben sind nicht trivial reparametrisierbar, was ihre Verwendung für VAEs und Posterior-Inferenz erschwert. Die Verwendung einer Gumbel-Softmax/Concrete Relaxierung der kategorischen "Option" würde jedoch eine multimodale, reparametrisierbare Verteilung liefern.
- Autoregressive Faktorisierungen von Policy-/Wertverteilungen. Insbesondere kann die kategorsche Verteilung jede diskrete Verteilung modellieren.
- In RL kann man dies ganz vermeiden, indem man die Symmetrie der Wertverteilung durch rekurrente Policies, Rauschen oder verteilungsbezogene RL bricht. Dies hilft, indem die komplexen Wertverteilungen in jedem Zeitschritt in einfachere bedingte Verteilungen zerlegt werden. 
- Lernen mit energiebasierten Modellen, d.h. ungerichteten grafischen Modellen mit Potenzialfunktionen, die auf eine normalisierte probabilistische Interpretation verzichten. Hier ist ein Beispiel für diese Anwendung auf RL.
- Normalizing Flows: Hier lernen wir invertierbare, volumenverfolgende transformationen von Verteilungen, die wir leicht manipulieren können.

Sehen wir uns den letzten Ansatz an - Normalizing Flows.

## Substitution und Volumenveränderung

Wir wollen eine gewisse Intuition entwickeln, indem wir die linearen Transformationen von 1D-Zufallsvariablen untersuchen. $$X$$ sei die Verteilung $$Uniform(0,1)$$. Sei die Zufallsvariable $$Y=f(X)=2X+1$$. $$Y$$ ist eine einfache affine Transformation (Skalierung und Verschiebung) der zugrundeliegenden "Ausgangsverteilung" $$X$$. Das bedeutet, dass eine Stichprobe $$x_i$$ aus $$X$$ in eine Stichprobe aus $$Y$$ umgewandelt werden kann, indem einfach die Funktion $$f$$ darauf angewendet wird. 

![Tux, the Linux mascot](/assets/images/blog/volumechange)



## Closing thoughts

## Credits

<span>Photo by <a href="https://unsplash.com/@jjying?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">JJ Ying</a> on <a href="https://unsplash.com/?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

*[SERP]: Search Engine Results Page
