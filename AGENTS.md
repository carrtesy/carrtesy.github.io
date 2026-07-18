# Dongmin Kim - Personal Portfolio Website

## Project Overview

GitHub Pages personal portfolio, replacing the previous Google Sites.
Live URL: **https://carrtesy.github.io/**

## Tech Stack

- **Frontend**: Pure HTML/CSS/JavaScript (no frameworks)
- **Data**: JSON files per section — edit content without touching HTML
- **Hosting**: GitHub Pages (`carrtesy/carrtesy.github.io`, `master` branch)

## Design

Terminal (tmux) concept, amber-phosphor palette. The whole page is one terminal window:
sections render as shell commands (`whoami --verbose` — photo, name, title, and the
About text from `data/about.json` — `git log --career --graph`, `ls publications/`,
`ping dongmin`). Clicking a career
commit or publication opens a **viewer pane** — a tmux-style vertical split on wide
screens, a floating terminal-window modal on narrow/portrait screens. The viewer
replays a sequence of typed commands (`show … --story`, `mpv video`, `imgcat photo`,
`open paper.pdf`, `curl … --preview`, `ls -l links/`), each followed by its output
(story chat bubble, video, image, embedded PDF, link-preview card, symlink-style
link list). `q` / `Esc` / backdrop tap closes the pane. Typing animation is skipped
under `prefers-reduced-motion`.

## File Structure

```
├── index.html              # Main page — terminal UI, renders all sections from JSON
├── style.css               # All styles (terminal theme, split pane / modal viewer)
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

## Schema Additions (terminal UI)

- `data/about.json` → optional `"highlights": ["anomaly detection", …]` — substrings of `text` rendered in amber
- `data/profile.json` → optional `"nickname": "tommy"` — used in `ping` output labels (`tommy's email: …`); falls back to first name
- `data/experience.json` / `data/education.json` → optional `"logo": "assets/img/logos/…png"` — shown at the far left of the `git log --career` row (falls back to a `*` commit node)

## Features

- **Viewer pane** (split / modal): click a career commit or publication → typed-command
  sequence renders story bubble, media (video/photo), embedded arXiv PDF, and links
- **Command mapping**: story → `show <slug> --story`; video → `mpv <file> --loop`;
  image → `imgcat <file>`; PDF → `open <slug>.pdf` (with ⛶ fullscreen + open-in-new-tab
  toolbar); experience link or publication `blogUrl` (when no PDF) → `curl … --preview`
  (microlink.io preview card, for X-Frame-Options-blocked pages); links → `ls -l <topic>/links/`
  (e.g. `ls -l aegis/links/` — symlink-style list from `links[]`, falling back to
  `pdf`/`blogUrl` when `links` is empty)
- **Theme toggle**: button in the terminal titlebar switches dark ↔ light terminal
  palettes (`data-theme` on `<html>`, persisted in localStorage; dark is default)
- **First-author badge**: derived from `authors` markup — `<strong>` first → `1st author`,
  `<strong>…*</strong>` → `co-1st`
- **Fake commit hashes**: deterministic hash of title+organization; first career entry
  whose date contains "Present" gets `(HEAD → now)`
- **Responsive**: viewer is a tmux split ≥880px wide, a floating terminal modal below
  (or portrait ≤1080px); tmux status bar tracks open windows

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
