{% extends 'markdown.tpl' %}

{%- block header -%}
---
title: "{{resources['metadata']['name']}}"
date: 2021-99-99
last_modified_at: 2021-99-99
categories:
 - To
 - Be
 - Written
tags:
 - Need
 - Be
 - Modified
 
use_math: true
---
{%- endblock header -%}

{% block stream %}	
```text
{{ output.text }}
```
{% endblock %}

{% block data_text %}
```text
{{ output.data['text/plain'] }}
```
{% endblock %}

{% block traceback_line %}
```text
{{ line | strip_ansi }}
```
{% endblock %}
