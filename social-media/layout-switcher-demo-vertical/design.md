# Design system — Path-finder demo

Tactical / mission-control aesthetic, mirroring the live site at uapdisclosuremirror.com.

## Palette

| Token | Hex       | Usage                                |
|-------|-----------|--------------------------------------|
| bg    | `#05070A` | Composition background               |
| fg    | `#cbd5e1` | Body text                            |
| dim   | `#7c8aa1` | Subtle labels                        |
| cyan  | `#57c7ff` | Primary accent — UI, active path     |
| amber | `#ffd34d` | Secondary accent — separators, CTA   |
| white | `#f3f4f6` | Path-node rings (selection)          |

## Agency colors (used in graph nodes)

| Agency             | Hex       |
|--------------------|-----------|
| Department of War  | `#ff6b6b` |
| FBI                | `#ffd34d` |
| NASA               | `#57c7ff` |
| Department of State| `#b07cff` |

## Typography

- Single family: `ui-monospace, "JetBrains Mono", SFMono-Regular, Menlo, Consolas, monospace`
- All headings + body + UI in monospace (the whole video reads as terminal output)
- Letter-spacing 0.02em on display, 0.06em on small caps labels
- Title 96px, sub-title 18px (small caps), body 32px, captions 24px, fine print 14px

## Motion

- Snappy: `power4.out` for HUD reveals
- Smooth: `power2.out` for fades + slides
- Linear: typewriter effects
- Slight bounce: `back.out(1.2)` for stagger entrances
- Dreamy: `power3.inOut` for outro

## Density / scale

- Outer padding 80px on all scenes
- Generous negative space — this is "tactical archive", not "full screen of stuff"
- A tiny scanline overlay on top (CSS gradient) to reinforce monospace/terminal feel

## What NOT to do

- No gradients spanning full width on dark backgrounds (banding)
- No emoji / no playful styling
- No multicolor backgrounds — black or near-black only
- No marketing language in captions; keep voice technical and direct
