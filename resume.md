---
layout: page
title: Kieran Didi
description: Senior Machine Learning Scientist at NVIDIA and PhD student in Computer Science at Oxford.
hide_description: true
buttons:
  print: true
  pdf: /assets/CV_Kieran_Didi.pdf
  json: /assets/resume.json
---

{% assign cv = site.data.resume %}
**{{ cv.basics.label }} · {{ cv.basics.location.city }}, {{ cv.basics.location.countryCode }}**

{{ cv.basics.summary }}

{% include pro/resume/buttons.html %}

## Education

{% for degree in cv.education %}
**{{ degree.studyType }} in {{ degree.area }}, {{ degree.institution }}** · {{ degree.startDate | slice: 0,4 }}–{{ degree.endDate | slice: 0,4 }}{% if degree.gpa and degree.gpa != '/' and degree.gpa != empty %} · {{ degree.gpa }}{% endif %}
{% for course in degree.courses %}- {{ course }}
{% endfor %}
{% endfor %}

## Work experience

{% for job in cv.work %}
**{{ job.position }}, [{{ job.company }}]({{ job.url }})** · {{ job.location }} · {{ job.startDate | slice: 0,7 }}–{% if job.endDate == empty %}present{% else %}{{ job.endDate | slice: 0,7 }}{% endif %}

{{ job.summary }}
{% for highlight in job.highlights %}- {{ highlight }}
{% endfor %}
{% endfor %}

## Research publications

[Publications with author lists and project resources](/publications/)

{% for pub in cv.publications %}- {% if pub.url != empty %}[{{ pub.name }}]({{ pub.url }}){% else %}{{ pub.name }}{% endif %}. {{ pub.publisher }}{% if pub.releaseDate != empty %}, {{ pub.releaseDate }}{% endif %}.
{% endfor %}

## Teaching

{% for item in cv.teaching %}- {{ item }}
{% endfor %}

## Selected talks and presentations

{% for item in cv.presentations %}- {{ item }}
{% endfor %}

## Scholarships and awards

{% for award in cv.awards %}- {{ award.date }} · {{ award.title }}
{% endfor %}

## Voluntary experience

{% for role in cv.volunteer %}
**{{ role.position }}, [{{ role.organization }}]({{ role.url }})** · {{ role.startDate | slice: 0,7 }}–{% if role.endDate == empty or role.endDate == nil %}present{% else %}{{ role.endDate | slice: 0,7 }}{% endif %}

{{ role.summary }}
{% endfor %}
