# Profile card — text-to-image prompt template

Card art for the app's music-panel selection grid (`assets/profiles/<id>.jpg`). The cards
sit on a near-black UI (`#0a0b0f`); each shows a **title + subtitle at the bottom-left in
white** and up to three small badges at the top-right, over a dark bottom-to-top shade the
app draws itself. Art therefore has to be **dark, calm, and quiet in the lower third** so
that overlaid white text stays legible — it sets a mood, it is not the focus.

Keep one coherent look across every card: they read as a set, not a stock-photo grab bag.

## Output spec

- **Aspect ratio 2:1**, landscape. The card renders ~2.1:1 with `background-size: cover`.
- **Size: 1024 × 512 px** (generate at 2× — `2048 × 1024` — if the model allows, then
  downscale; keeps edges clean on high-DPI phones).
- **Format: JPG**, quality ~85, sRGB. Aim < 250 KB per card (they're downloaded on the
  phone). No alpha.
- **File name = the profile id**, e.g. `quick_focus.jpg`, then set `"image":
  "/profile/quick_focus.jpg"` in `profiles.json`.

## House style (prepend to every prompt)

```
Abstract atmospheric background, dark moody minimal, soft gradients and gentle light,
cinematic depth, fine film grain, elegant and modern, deep near-black base (#0a0b0f)
with {{ACCENT}} light. Low contrast in the lower-left third so it stays dark and uncluttered.
No text, no words, no letters, no logos, no watermark, no people, no faces, no UI, no icons.
Wide 2:1 composition, subject drifting to the upper-right, empty calm space lower-left.
```

- `{{ACCENT}}` — the card's accent colour, taken from its `gradient` in `profiles.json` so
  the art and the fallback gradient agree. The two house accents are **indigo `#7c8cff`**
  (focus / energy) and **teal `#4fe3c1`** (calm / nature); warmer ambers are fine for
  sleep/nap.

## Per-card fill

Append a short subject line describing the profile's mood. Keep it abstract (textures,
light, atmosphere) — avoid literal objects that fight the title.

| id              | accent            | subject line to append                                                        |
| --------------- | ----------------- | ----------------------------------------------------------------------------- |
| `quick_focus`   | indigo `#7c8cff`  | crisp directional light beams, a sense of alert momentum, cool blue-violet     |
| `eeg_flow`      | violet `#7d4fe3`  | flowing neural filaments and soft waveforms, adaptive electric violet          |
| `deep_relax`    | teal `#4fe3c1`    | slow soft tidal gradients, warm-dark teal, exhaling calm                       |
| `nature_relax`  | green `#2e7d4f`   | misty forest light and drifting particles, deep green, unhurried               |
| `sleep`         | deep blue `#2b3a7a` | near-black night sky, faint stars, very low light, restful                    |
| `meditation`    | slate blue `#3a5a8a` | still water and a single soft horizon glow, centered breath, minimal         |
| `afternoon_nap` | amber `#8a5a3a`   | warm hazy late-afternoon sun, soft dust in light, cozy amber                    |
| `flow_work`     | indigo `#5b6bff`  | long steady light trails, immersive tunnel of gentle motion, deep blue         |
| `caffeine_free` | teal-green `#2e7d6a` | clear cool dawn light, awake but unhurried, fresh teal-green                  |

`manual` needs no art — it uses its neutral gradient and a dashed border in the UI.

## Negative prompt (append to every prompt)

```
text, words, letters, typography, logo, watermark, signature, people, person, face, hands,
UI, buttons, icons, frame, border, bright center, high contrast, busy clutter, harsh light,
oversaturated, cartoon, lowres, jpeg artifacts
```

## Worked example (`quick_focus`)

```
Abstract atmospheric background, dark moody minimal, soft gradients and gentle light,
cinematic depth, fine film grain, elegant and modern, deep near-black base (#0a0b0f) with
indigo #7c8cff light. Low contrast in the lower-left third so it stays dark and uncluttered.
No text, no words, no letters, no logos, no watermark, no people, no faces, no UI, no icons.
Wide 2:1 composition, subject drifting to the upper-right, empty calm space lower-left.
Crisp directional light beams, a sense of alert momentum, cool blue-violet.
--ar 2:1
Negative: text, words, letters, typography, logo, watermark, signature, people, person,
face, hands, UI, buttons, icons, frame, border, bright center, high contrast, busy clutter,
harsh light, oversaturated, cartoon, lowres, jpeg artifacts
```
