# The Archive — a full front-end makeover

> **Live mockup:** `design-proposals/archive-mockup.html`. Open it in a
> browser to feel the design — it uses real records from the corpus.

---

## What the current site looks like, and why I'd change it

The site today reads as a generic dev-tool: dark gray background, mono
font everywhere, blue/purple accent, dense rows of metadata. It gestures
at the tactical-command-centre aesthetic the README hints at, but it
never commits. Every screen looks like a Linux dashboard.

The content deserves more. This is the only public mirror of the war.gov
UFO release — declassified intelligence, FBI case files, NASA radio
logs, witness composite sketches of "orange orbs launching red orbs."
The visual language should treat that material with the gravity it
has, not the gravity of an admin panel.

## The direction — "The Archive"

A declassified-records editorial aesthetic. Warm dark "banker's-lamp"
surfaces, cream-paper detail folders, serif headlines, monospace
metadata, real classification banners, and stamp-red accents used only
for the things that genuinely deserve emphasis. It reads like an
investigative magazine spread that someone has just pulled the binder
clip off of.

### The three fonts

| Job | Font | Why |
|---|---|---|
| Editorial display + body | **Newsreader** (variable opsz) | A magazine serif with optical sizes — looks like a long-form feature, not a blog. Holds up at 96 px AND at 14 px. |
| Technical metadata | **JetBrains Mono** | Distinct from the generic system mono, with the right amount of character at small sizes. Used only where mono actually helps (hashes, dimensions, dates, IDs). |
| Stamps + tactical labels | **Big Shoulders Display** (700/900) | Condensed industrial sans — the typeface of a rubber-stamp banner. Used **only** for classification headers, agency chips, and the briefing labels. Never in body copy. |

No Inter. No system sans. No Space Grotesk. Three faces, three jobs,
no overlap.

### The palette

```
bg-deep    #14110c   /* aged document ink — the dominant surface */
bg-mid     #1c1813   /* slightly lighter panel surface */
paper      #ecdfca   /* warm cream — the "lit" surface, detail folders */
ink        #16110a   /* the type on cream */
cream      #e9dec3   /* type on dark surfaces */
stamp      #9a2c2c   /* classification-stamp red — used sparingly */
amber      #c89a3a   /* highlighter / 'derivative' warning */
olive      #6c7438   /* document tag accent */
cyan       #4f7886   /* secondary tag accent */
```

The contrast couple is dark-olive-ink vs warm-cream-paper. The stamp
red is rationed — it only appears where the meaning actually warrants
it (active filter, edit signature, "DECLASSIFIED" stamp).

### Signature moves

1. **The classification banner** running edge-to-edge at the top
   and bottom of every page, in condensed caps, in stamp red. Reads
   like the cover sheet of the briefing.

2. **The wordmark.** "The *Disclosure* Archive." Set in 96 px Newsreader
   with the middle word italic and amber, terminating in a stamp-red
   period. One thing on the page, big and confident.

3. **The folder-tab record card.** Each search hit is a panel with:
   a coloured page-corner tab encoding the type (cyan PDF / olive IMG
   / amber VID), the agency as a coloured stamp, the title in serif,
   an italic blurb in the magazine voice, and a small *sepia-toned*
   thumbnail with subtle scanlines. The thumbnail looks like it was
   slid into the folder, not pasted into a dashboard.

4. **The opened folder (detail panel).** Renders in cream, with a
   rotated red `DECLASSIFIED` stamp at the top, a typewriter-style
   ID line, an FBI-302-style form grid, a justified serif extract
   with a small-caps lede, and a torn dog-ear corner showing the
   dark surface beneath. Real-feeling document.

5. **Stamps with bracket corners** for filters and tags. They read
   as physical stamps, not buttons. Active state recolours the whole
   stamp body (stamp-red for "all", cyan for "case files", olive for
   "images", amber for "videos"). Type-aware so the colour language
   is consistent throughout.

6. **Page-corner type indicators** on every record card — instant
   visual grammar for what you're looking at, no icon library needed.

7. **Real redaction.** The `REDACTED` chip is a stamp-style overlay,
   and inline redactions in the detail panel render as `inline-block`
   ink-on-ink boxes that hide their content. Subtle scratch on the
   stamp to read as actually printed.

8. **Grain + sepia.** A faint SVG film-grain overlays the whole
   document; thumbnails get a light sepia + scanline treatment.
   No more "screenshot of a JSON viewer."

9. **Staggered reveal on load.** Records come in with a 70-ms-stepped
   fade+lift, the detail folder slides up half a second later. One
   well-orchestrated load instead of nothing.

## What this changes, view by view

| View | Current | Proposed |
|---|---|---|
| Search header | Dark bar, system mono input, blue accent | Big editorial wordmark + typewriter-style search on cream + stamp-red counter |
| Filter chips | Dark pill buttons with counts | Bracket-cornered stamps, type-aware colour, dotted-rule separators |
| Result row | Dense row, thumb left, blue tag chips | Folder-tab card with corner colour, agency stamp, serif title, italic blurb, sepia thumb |
| Detail panel | Same dark surface, fields stacked | Lit cream "opened folder" with rotated DECLASSIFIED stamp, form-grid metadata, justified serif extract, dog-eared corner |
| Lightbox | Black overlay, monospace toolbar | Same olive-document surface, classification stripe, paper-coloured forensics panel |
| Map / Timeline / Graph | Library default with dark theme | Briefing-room frame: corner brackets, scope rings, the same fonts and palette so the chrome reads as one site |
| Globe / vault entry | Plain links | Treated as briefing modes ("CARTOGRAPHIC", "CHRONOLOGICAL", "GRAPHICAL") in the masthead, condensed-caps |
| Empty / error states | Plain text | "FILE NOT FOUND" stamp, file-folder illustration where appropriate |

## Scope, honestly

This is a real piece of work — not "swap a stylesheet". Concretely:

**Phase 1 — chrome only (~1 day).** Drop in the fonts, recolour the
palette tokens, replace the wordmark + classification banners, restyle
the filter bar as stamps. ~200 lines of CSS, no functional changes.
Lowest-risk visual upgrade. Worth doing even if Phase 2/3 never ships.

**Phase 2 — search + detail (~2-3 days).** Reskin the search results
and detail panel to the folder-card / opened-folder layouts. Some HTML
restructuring in `renderRow()` and `showDetail()`, but the data shape
doesn't change. This is where the design actually starts to feel real.

**Phase 3 — viewers (~2-3 days).** Apply the same chrome to the PDF
viewer, lightbox, map, timeline, graph, vault entry. Mostly chrome work
again — the underlying widgets (Leaflet, vis-timeline, Cesium) keep
their behaviour; we restyle their containers.

**Phase 4 — polish (~1-2 days).** Empty/error states, motion polish,
ensure the redacted-content treatment is consistent everywhere,
audit dark-on-cream contrast for accessibility.

Total: **~6–9 focused days** for a complete overhaul, with Phase 1
worth shipping on its own as an inexpensive immediate refresh.

## What I deliberately did NOT do

- **Cult-aesthetic.** No X-Files VHS glitch, no neon green Matrix
  rain, no flying-saucer cursor. The material is serious and the
  framing principle ("not a verdict, show your work") demands a
  serious visual register too. Anything kitschy would undermine the
  credibility you've spent the last year building.

- **Cyberpunk dashboard.** Tempting because there's a tactical-HUD
  read of the same material, but every UAP site does that already.
  Editorial-archive stands out.

- **Light theme.** The dark substrate is essential — it makes the
  cream "opened folder" feel lit, and it's the right register for
  late-night reading of declassified material. A light mode could
  be added later (literally invert ink ↔ paper for the dark/light
  swap) but isn't a Phase 1 concern.

## Risks / trade-offs you should know about

- **Three custom font families** = roughly 200-300 KB of font payload
  on first load. Self-host with `font-display: swap` and a `preload`
  on the Newsreader subset and this is invisible. We're not adding
  a framework — the chrome is still vanilla HTML/CSS.

- **The cream detail panel** is a major contrast shift from the
  surrounding dark UI. Some visitors will find it visually loud the
  first time. The editorial version is right because it's *meant* to
  feel like a document being read, not a side panel. Worth A/B-ing
  a softer-cream variant if you get complaints.

- **Big Shoulders Display in all caps** + the heavy stamp red is
  loud. The mockup uses it sparingly on purpose — only banners, only
  filter stamps, only briefing labels. If we let it leak into body,
  it'd become a shouting site.

- **Pure-CSS folder-tab cards** are cheap to ship but get visually
  repetitive on long result lists. Mitigation: the per-type page-corner
  colour gives 4 distinct "moods" within the column, and the thumbnail
  variance keeps it alive. If it still feels monotone after live use,
  add a tiny vertical variation (taller cards for IMG / VID).

## Next step

Open `archive-mockup.html`. If the direction feels right, I'll start
Phase 1 (tokens + classification banners + wordmark) as a small,
revertable PR you can leave on for a day to feel it in production
before committing further.

If a different direction feels right — too editorial, want more
tactical, want the X-Files cult thing despite my objection — I can
build a second mockup with one tweak: **the same content laid out
under a different aesthetic**. Easier to compare two mockups than to
argue an abstract direction.
