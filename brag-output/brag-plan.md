# Brag Plan: Enterprise Research Intelligence

## What is this app?
Enterprise Research Intelligence is a FastAPI RAG engine: you upload a PDF, it
chunks and indexes it (BM25 sparse + dense embeddings fused by reciprocal-rank
fusion, reranked to a final context), and you ask questions — getting back an
answer where every claim is wired to a citation, a grounding check, a
confidence score, and a failure analysis. It verifies its own answers against
the source pages, and tells you when it can't.

## The angle
A quiet, premium engineering film. No hype — the product's confidence IS the
joke and the payoff: the system that will not let an answer exist unless it can
point at the page it came from. The visuals are the real product: the three-step
panel flow, then the answer panel with its "Grounded" badge, confidence pill,
and citation cards. It feels like the demo you'd see from a serious research
infrastructure team.

## Hook (first 2-3 seconds)
The product name as eyebrow — "ENTERPRISE RESEARCH INTELLIGENCE" — under a
single clean line: "Ask questions about your documents." The subtitle promises
evidence-backed citations. Plain gray-on-white hero, exactly like the site.

## Key moments (the middle)
- **The three-step flow.** "1 · Upload document → 2 · Process document →
  3 · Ask a question." Three white panel cards arrive one by one.
- **The working answer panel.** A real question — "What changed in the
  company's product strategy?" — with an answer carrying clickable [C1][C2]
  citation links, a "Grounded · 3/3 claims supported" badge, and a
  "Confidence 0.940" pill.
- **The evidence.** Three citation cards (C1, C2, C3) drop in: source file,
  "Page N", and an excerpt. Proof, stacked on screen.

## Outro / punchline
Two hard lines, then the logo:
"Answers must come from somewhere." → "Now they say where."
Then: ENTERPRISE RESEARCH INTELLIGENCE — "Evidence-backed answers, page by
page."

## User flow worth showing
**Entry → key action → result:**
1. Upload a document (the flow starts at the three-step cards; the site's own
   step names are shown).
2. Process and index it — "Ready. Indexed N chunks" is implied by the answer
   panel being unlocked.
3. Ask a question → grounded answer + confidence + citation cards appear.

The centerpiece scenes (2-3) recreate the working app: the step panels and the
question→answer→evidence result view.

## Tone
- Preset: polished
- Creative direction: quiet premium product film — serious infrastructure,
  confident restraint
- Interpretation: third person, no voice; 4 scenes, 4-6s holds; slow
  crossfades; generous letter-spacing; nothing aggressive. The system's
  certainty is the drama.

## Format: landscape — 1920x1080
## Duration: 19.6 seconds

## Visual identity (from the project)
- Background: #f3f4f6 (the site body) with a deeper #111827 field for the
  hook and outro
- Accent: #111827 (button black) + #374151 hover; status text gray #6b7280
- Text: #1f2937 on white cards
- Display font: Inter (bold, tight) — the site's font
- Body font: Inter
- Strongest visual element: the white rounded panel cards with the numbered
  eyebrow steps and the confidence pill + citation cards

## Share copy (draft)
Upload a document. Ask a question. Get an answer where every claim points
back to the exact page it came from. Enterprise Research Intelligence.

## Audio direction
- Role: sparse professional accents — a warm clean bed, minimal cueing, the
  product carries the film
- Music: happy-beats-business-moves-vol-12-by-ende-dot-app.mp3 (steady,
  clean, 109.96 BPM)
- Music treatment: start at 0, bed volume ~0.3, gentle fade-out over the final
  logo
- Music cue guidance: bundled preset read
  (assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.md).
  Strong cues in window: 8.74, 10.93, 13.11, 17.47, 18.56. Use at most 1-3
  beat-locked majors (hook line land ≈ 1.6s→nearest beat 1.64; answer reveal
  ≈ 8.6s→nearest strong 8.74; third citation card ≈ 13.0s→nearest strong
  13.11). Sequential step/citation cards snap to nearby beats but honor the
  reading floor — ~1.6s apart (roughly every 3 beats at this tempo) is right
  for readable sequential text.
- Audio-reactive treatment: subtle; let music RMS/bass gently breathe the
  hero glow and the answer-panel card presence. No waveform/equalizer visuals.
- SFX posture: sparse and restrained — one soft reveal hit per card group, a
  single bell on the answer payoff, a low click on simulated Ask interaction
- Audio-coupled moments:
  - Step cards — soft drop / card sound on each arrival
  - Ask interaction — click on the simulated Ask press
  - Answer grounding payoff — one tasteful bell
  - Citation cards — soft card sounds (accent only)
  - Final logo — a low bell that rings over the fade
- Restraint rule: two sounds may not overlap; never louder than the bed feels
  natural; no whooshes, no glitches, no comedy.

## Storyboard

### Scene 1 — Hook — 3.8s
Deep near-black field (#111827). Eyebrow "ENTERPRISE RESEARCH INTELLIGENCE"
types/fades in top center; headline "Ask questions about your documents."
settles below it; the subtitle "Evidence-backed citations, from the page to
the claim." fades up; a subtle white glow breathes with the music.
Sequential/interaction: yes — eyebrow, headline, subtitle appear one by one
(≈1.2s apart), each held to the reading floor.
Audio intent: calm, deliberate open.
Audio-coupled idea: subtle glow breathing with RMS; soft drop as the headline
lands (≈1.6s).
Music: intro of vol-12.
Transition mood: slow crossfade with a slight scale (scene enters 0.98→1.0) → Scene 2

### Scene 2 — The three steps — 4.9s
Light gray page (#f3f4f6). Three white rounded panel cards arrive one by one
in a vertical stack: "1 · Upload document", "2 · Process document",
"3 · Ask a question". Each card has a one-line caption underneath its label
("PDFs, docs, markdown", "Chunk & index into a hybrid index", "Grounded,
citation-backed answers"). Cards land ≈1.6s apart on beats; all stay on screen
until the transition.
Sequential/interaction: yes — 3 cards one by one, each with a soft card sound.
Audio intent: steady, confident build.
Audio-coupled idea: card-per-beat-group arrival sounds.
Transition mood: crossfade → Scene 3

### Scene 3 — The working answer — 6.9s
The real app's answer panel on white. Question in the shaded header:
"What changed in the company's product strategy?" — the Ask button gets a
simulated click — then the answer types/assembles in:
"The product strategy shifted toward enterprise customers, broader platform
capabilities, and expanded monetization. [C1][C2]" Above the answer, a
"Grounded · 3/3 claims supported" badge; beside it, a "Confidence 0.940" pill.
Then three citation cards (C1, C2, C3) drop in together-fast (~0.4s apart),
each showing the source file, "Page N", and a short excerpt.
Sequential/interaction: yes — Ask click, answer text, badge+pill, then
citation cards.
Audio intent: the payoff — precise, soft satisfaction.
Audio-coupled idea: click on Ask; bell on the grounded badge; soft card sounds
on citation cards.
Transition mood: crossfade → Scene 4

### Scene 4 — Outro — 4.0s
Back to the near-black field. Line 1: "Answers must come from somewhere."
Line 2 (rated a beat later): "Now they say where." The logo locks in:
"ENTERPRISE RESEARCH INTELLIGENCE" with the tagline "Evidence-backed answers,
page by page." Glow lingers; music fades; one low bell rings over the fade.
Sequential/interaction: yes — two lines then logo, ~1.3s apart.
Audio intent: close it with certainty.
Audio-coupled idea: bell on the logo lock; final fade to silence.
Transition mood: end card.

**Music mood for this video:** steady, confident, warm-clean.
**Audio summary:** a warm clean bed carries the whole film; sparse soft
accents land on the step cards, one click on Ask, one bell on the grounded
answer, low card sounds on evidence, and a single bell rings the logo in as
the music fades.