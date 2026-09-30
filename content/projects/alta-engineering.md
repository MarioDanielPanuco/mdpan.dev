+++
title = "alta.engineering"
description = "Marketing site for my construction company: a Dioxus fullstack app pre-rendered to static HTML with WASM hydration, a WebGL wireframe-house hero, and a Cloudflare Worker behind the contact form."
weight = 1
date = "2026-09-30"
[taxonomies]
tags = ["rust", "dioxus", "wasm", "cloudflare", "projects"]
[extra]
local_image = "images/alta-engineering.png"
+++

[alta.engineering](https://alta.engineering) is the site for Alta
Construction & Engineering Inc., a Santa Clara general contractor doing residential
foundations, framing, and public-works concrete.

## Stack

- **Dioxus 0.7 fullstack, pre-rendered to static HTML.** Every route is
  rendered at build time (`dx build --ssg`) and shipped as plain HTML, then a
  WASM bundle hydrates it in the browser. First paint depends on none of the
  WASM, JS, or fonts.
- **SVG "drawing set" design language.** The page
  reads like a construction drawing: benchmarks, station lines, section
  callouts.
- **A three-d WebGL hero** that assembles a wireframe house in the browser.
- **Cloudflare Workers hosting.** Static assets serve the pages; a small
  TypeScript Worker handles the contact form.
- Build-time SEO output: `sitemap.xml`, `robots.txt`, `llms.txt`, and
  `GeneralContractor` JSON-LD generated from the same route list as the pages.

## Footprint

Release build, measured on the deployed artifacts:

| Asset                          | Size                        |
| ------------------------------ | --------------------------- |
| WASM (includes the WebGL hero) | 1.13 MB raw, 460 KB gzipped |
| JS glue                        | 17 KB gzipped               |
| CSS                            | 5.8 KB gzipped              |
| Fonts (all faces)              | 242 KB total                |

Source is private. Email me if you want details on the build pipeline or
the SSG setup.
