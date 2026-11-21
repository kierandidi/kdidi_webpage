---
layout: welcome
title: Publications
---

[Download the publication list (PDF)](/assets/Publication_List_Kieran_Didi.pdf) · [Download my CV (PDF)]({{ '/assets/CV_Kieran_Didi.pdf' | relative_url }})

\* Equal contribution.

{% include publication-list.html %}

## Selected Talks and Presentations

{% for item in site.data.resume.presentations %}- {{ item }}
{% endfor %}

## Reviewing

I have completed at least 34 peer reviews, including 31 conference and workshop submissions and three journal manuscripts.

- **Journals** · Nature Biotechnology, Nature Communications, and GENETICS.
- **Conferences** · AAAI 2027, ICML 2026, ICLR 2026, NeurIPS 2026 Evaluations and Datasets, and NeurIPS 2025 Datasets and Benchmarks.
- **Workshops and symposia** · ICLR GEM (2024, 2026), ICML GenBio 2025, ICML CompBio 2023, NeurIPS AI4Science 2023, NeurIPS GenBio 2023, and MoML at MIT 2023.

## Teaching

{% for item in site.data.resume.teaching %}- {{ item }}
{% endfor %}
