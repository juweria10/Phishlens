# PhishLens design system

Reference: DESIGN.md (editorial serif + humanist sans, hierarchical radii, dark "product surface" for evidence).
Used as inspiration, not copied: the palette is cool and security-focused instead of cream and coral.

## Idea
The result is an inspection. A dark panel holds the verdict, a risk scale and the URL broken into its parts
(scheme, subdomain, domain, ending, path), so the user sees *where* the evidence is. Everything else stays quiet on a pale canvas.

## Colour
| Token | Hex | Use |
|---|---|---|
| canvas | #F1F3EF | page background |
| paper | #FAFBF8 | inputs |
| ink / body / muted | #0E1A19 / #2B3836 / #5E6B68 | text |
| rule | #D3D9D2 | dividers |
| night | #0C1716 | sidebar and result panel |
| brand | #0F6B63 | primary action, focus |
| danger / safe | #D2402A / #2A8559 | phishing / legitimate only |
| warn | #A9690F | low reliability, cautions |

Red and green are reserved for verdicts and are never the only signal (labels and bar position always accompany them).

## Type
- Newsreader (serif): page titles, verdict, big numbers
- IBM Plex Sans: interface text
- IBM Plex Mono: URLs and numeric values

## Layout rules
- Sidebar navigation with a marked active item; content max width 1120px.
- Cards are used once, for the result. Other sections use rules and whitespace.
- Radii: 8px controls, 12px result panel, pill for verdict tags.
- Links are never clickable in results. Motion: one short fade on the result panel, disabled when reduced motion is requested.
