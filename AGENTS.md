# Dongmin Kim - Personal Portfolio Website

## Project Overview

A GitHub Pages-based personal portfolio website, replacing the previous Google Sites (https://sites.google.com/view/tommy-dm-kim/about-me).

## Tech Stack

- **Frontend**: Pure HTML/CSS/JavaScript
- **Design**: Minimal academic style
- **Data**: JSON-based content management (easy to modify)
- **Hosting**: GitHub Pages

## File Structure

```
carrtesy/
├── index.html          # Main page (dynamic JSON loading)
├── style.css           # Stylesheet (responsive)
├── data.json           # Content data (editable)
├── AGENTS.md           # Project documentation
└── assets/
    └── img/
        ├── profile.jpg
        └── papers/     # Publication images
            ├── AAAI26_AEGIS.png
            ├── ICLR26_ReTabAD.png
            ├── AAAI25_Ratab.png
            ├── AAAI24_M2N2.png
            ├── PKDD23_DeepImbTS.png
            ├── NeurIPS22_WaveBound.png
            └── CIKM22_ResCAL.png
```

## Data Structure (data.json)

To modify content, simply edit the `data.json` file.

### Profile
```json
{
  "profile": {
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
}
```

### Publications
```json
{
  "title": "Paper title",
  "venue": "Conference/Journal Year",
  "authors": "Author list (use <strong>Name</strong> for emphasis)",
  "image": "assets/img/papers/image.png",
  "links": [
    { "label": "arXiv", "url": "https://..." },
    { "label": "Code", "url": "https://..." }
  ]
}
```

### Education (with optional link)
```json
{
  "date": "2021 - 2023",
  "title": "M.S. in Artificial Intelligence",
  "organization": "KAIST, DAVIAN Lab",
  "link": "https://davian.kaist.ac.kr/people"
}
```

## Request History

### 1. Initial Setup
- [x] Create GitHub Pages portfolio website
- [x] Reference existing Google Sites content
- [x] Use pure HTML/CSS
- [x] Minimal academic style design

### 2. Content
- [x] Write content in English
- [x] Reflect accurate publication list from Google Scholar
- [x] Add links for each paper (arXiv, Code, Project, Dataset, etc.)
- [x] About: Mention interest in developing Industrial Agentic AI

### 3. Contact/Links
- [x] Email: tommy.dm.kim@gmail.com
- [x] LinkedIn: https://www.linkedin.com/in/dongmin-kim-7056aa182/
- [x] Google Scholar: https://scholar.google.com/citations?user=kXKN8DwAAAAJ&hl=en
- [x] Remove Twitter

### 4. Images
- [x] Add images for each publication
- [x] Organize image folder: `assets/img/papers/`
- [x] Maintain image aspect ratio (prevent cropping)

### 5. Additional Links
- [x] AEGIS paper: Blog link (https://www.lgresearch.ai/blog/view?seq=633)
- [x] DAVIAN Lab link (https://davian.kaist.ac.kr/people)

### 6. Structural Improvements
- [x] Separate content into JSON (data.json)
- [x] Update year to 2026

## Local Development

```bash
# Start local server
python3 -m http.server 8000

# Open in browser
open http://localhost:8000
```

## Deployment (GitHub Pages)

```bash
git add .
git commit -m "Update portfolio"
git push
```

GitHub → Settings → Pages → Source: `main` branch
