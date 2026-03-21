# Dongmin Kim - Personal Portfolio Website

## Project Overview

GitHub Pages personal portfolio, replacing the previous Google Sites.
Live URL: **https://carrtesy.github.io/**

## Tech Stack

- **Frontend**: Pure HTML/CSS/JavaScript (no frameworks)
- **Data**: JSON files per section — edit content without touching HTML
- **Hosting**: GitHub Pages (`carrtesy/carrtesy.github.io`, `master` branch)

## File Structure

```
├── index.html              # Main page — renders all sections from JSON
├── style.css               # All styles (responsive, modal, link preview)
├── AGENTS.md               # This file
├── data/
│   ├── profile.json        # Name, title, email, social links, profile image
│   ├── about.json          # About Me text
│   ├── experience.json     # Work experience entries
│   ├── education.json      # Education entries
│   ├── publications.json   # Paper list
│   └── footer.json         # Footer year and name
└── assets/
    └── img/
        ├── profile.jpg     # Profile photo
        ├── undergrad.JPG   # Undergrad photo (education modal)
        ├── master.mp4      # Master's video (education modal)
        └── papers/         # Publication thumbnail images
            ├── AAAI26_AEGIS.png
            ├── ICLR26_ReTabAD.png
            ├── AAAI25_Ratab.png
            ├── AAAI24_M2N2.png
            ├── PKDD23_DeepImbTS.png
            ├── NeurIPS22_WaveBound.png
            └── CIKM22_ResCAL.png
```

## Data Schemas

### data/profile.json
```json
{
  "name": "Dongmin Kim",
  "title": "AI Research Scientist @ LG AI Research",
  "image": "assets/img/profile.jpg",
  "email": "tommy.dm.kim@gmail.com",
  "social": {
    "linkedin": "https://www.linkedin.com/in/dongmin-kim-7056aa182/",
    "github": "https://github.com/carrtesy",
    "scholar": "https://scholar.google.com/citations?user=kXKN8DwAAAAJ&hl=en"
  }
}
```

### data/publications.json
```json
[
  {
    "title": "Paper title",
    "venue": "Conference Year",
    "authors": "Author list (use <strong>Name</strong> to bold your name)",
    "image": "assets/img/papers/image.png",
    "pdf": "https://arxiv.org/pdf/...",        // null if not available
    "blogUrl": "https://...",                   // null if not available
    "story": "Personal comment shown in modal", // "" if none
    "storyDate": "Jan 1, 2025",                 // "" if none
    "links": [
      { "label": "arXiv", "url": "https://..." },
      { "label": "Code",  "url": "https://..." }
    ]
  }
]
```

### data/experience.json / data/education.json
```json
[
  {
    "date": "Aug 2023 - Present",
    "title": "AI Research Scientist",
    "organization": "LG AI Research",
    "link": "https://...",          // shown as [News] or [Website] in modal
    "media": "assets/img/...",      // image or video shown in modal body (education only)
    "mediaType": "image | video",   // education only
    "story": "Personal comment",
    "storyDate": "Mar 2026"
  }
]
```

## Features

- **Publications modal**: click a paper → left panel shows personal story comment, right panel shows PDF (arXiv) inline or a link preview card for external blog posts
- **Experience/Education modal**: same left story panel; experience shows link preview card, education shows image or autoplay video
- **Link preview card**: for URLs blocked by `X-Frame-Options` (LG AI Research pages), fetches OpenGraph metadata via `microlink.io` API and renders a thumbnail + title card that opens in a new tab
- **Expand icon** (↗): shown on list items that have modal content
- **Responsive**: mobile layout stacks story panel above content panel

## Local Development

```bash
python3 -m http.server 8000
open http://localhost:8000
```

> Must use a local server — `fetch()` is blocked on `file://` protocol.

## Deployment

Remote: `https://github.com/carrtesy/carrtesy.github.io.git` (`master` branch)

```bash
git add -A
git commit -m "Update portfolio"
git push origin master
```

Pages is served from `master` branch root (`/`).
