---
layout: page
title: Projects
---

<div class="featured-projects">
  {% for project in site.data.featured_projects %}
  <article>
    <a class="featured-project-link no-mark-external" href="{% if project.url contains '://' %}{{ project.url }}{% else %}{{ project.url | relative_url }}{% endif %}">
      <img class="project-image-{{ project.fit }}" src="{{ '/assets/img/projects/' | append: project.image | relative_url }}" alt="{{ project.alt | escape }}" decoding="async">
      <h2>{{ project.title | escape }}</h2>
    </a>
    <p>{{ project.description | escape }}</p>
  </article>
  {% endfor %}
</div>
