# Hyperframes Composition Brief: Enterprise Research Intelligence

## Objective
Create a short launch-style brag video for Enterprise Research Intelligence, a
PDF-upload RAG engine whose answers are citation-verified and grounding-checked.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 19.6 seconds

## Source Material
- Project root: `/home/gouthamx67/Downloads/enterprise-research-intelligence`
- Primary files read: `src/rag_engine/app/static/index.html`,
  `src/rag_engine/app/static/app.js`, `src/rag_engine/app/static/styles.css`,
  `src/rag_engine/app/question.py`, `scripts/final_demo.py`
- Product name: Enterprise Research Intelligence
- Tagline / strongest claim: "Ask questions about your documents" — answers
  carry evidence-backed citations, grounding validation, and confidence
- Key UI or visual moment to recreate: the 3-step panel-card flow and the
  question → answer → citation-card result view from the real frontend
- Copy that must appear verbatim:
  - "Ask questions about your documents."
  - Step labels: "1 · Upload document", "2 · Process document", "3 · Ask a question"
  - Answer: "The product strategy shifted toward enterprise customers, broader
    platform capabilities, and expanded monetization. [C1][C2]"
  - "Grounded · 3/3 claims supported"
  - "Confidence 0.940"

## Creative Direction
- Tone preset: polished
- Creative direction: quiet premium product film — serious infrastructure,
  confident restraint
- Interpretation: third person, no voice; 4 scenes, slow crossfades, generous
  letter-spacing, nothing aggressive. The system's certainty is the drama.
- Angle: The system will not let an answer exist unless it can point at the
  page it came from. Show the real product doing it.
- Hook: Near-black field, product eyebrow, "Ask questions about your
  documents." — deep calm, typed/fades in one by one.
- Outro / punchline: "Answers must come from somewhere." / "Now they say
  where." → logo + "Evidence-backed answers, page by page."
- Avoid:
  - Generic SaaS language ("streamline your workflow")
  - Abstract filler visuals and generic motion graphics
  - Unrelated visual redesign — keep the site's gray/white/black panel look

## Visual Identity
- Background: #111827 (hook/outro fields), #f3f4f6 (step/answer scenes)
- Text: #1f2937 on white cards; #6b7280 for labels/subtitles
- Accent: #111827 buttons, #374151 hover, #e5e7eb card borders
- Display font: Inter (bold), letter-spacing -0.02em for headlines; eyebrow
  uppercase with 0.12em tracking
- Body font: Inter (400/600)
- Visual references from the project: white rounded .panel cards (16px radius,
  soft shadow 0 8px 24px rgba(15,23,42,0.06)), the pill-shaped confidence
  badge, the citation cards with header (ID + source), "Page N" location, and
  excerpt

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. Hook — 3.8s — eyebrow, headline, subtitle appear sequentially on dark
2. The three steps — 4.9s — three white panel cards arrive one by one on gray
3. The working answer — 6.9s — question header + Ask click, grounded answer
   with [C1][C2] links, "Grounded · 3/3" badge + "Confidence 0.940" pill,
   then three citation cards with Page + excerpt
4. Outro — 4.0s — two punchlines then the logo on dark, music fades

Reading floors honored: sequential labels hold ≥0.8s settled; the answer
sentence (12 words) holds ≥3.6s; punchlines hold ≥1.3s each.

## Audio
- Audio role: sparse professional accents — warm clean bed, minimal cueing
- Audio arc: calm open → steady build → soft payoff → fade to one closing bell
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
  (vol-12, steady/clean, 109.96 BPM)
- Music treatment: bed volume ~0.3, gentle fade-out over the final logo
- Music cue guidance: bundled preset at
  `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.{json,md}`.
  Strong cues in window: 8.74, 10.93, 13.11, 17.47, 18.56. Suggest ≤3
  beat-locked majors (hook headline ≈1.6s→1.64; grounded-answer reveal
  ≈8.6s→8.74 strong; final citation card ≈13.0s→13.11 strong). Sequential
  text reveals snap to beats only where the reading floor is met (~1.6s apart
  ≈ every 3 beats); readability always wins.
- Audio-reactive treatment: subtle; music RMS/bass breathes the hero/logo glow
  and the answer-panel card presence. No waveform/equalizer visuals.
- Audio-coupled moments:
  - Scene 1 headline land — soft drop (≈1.6s)
  - Scene 2 step cards — soft card/drop sounds per arrival
  - Scene 3 Ask press — single click; grounded badge — one tasteful bell;
    citation cards — soft card accents
  - Scene 4 logo — low bell ringing over the music fade
- SFX selection guidance: low-HF-risk files for all card/UI moments
  (impact/impactSoft_medium_*, interface/drop_*, ui/click2-level clicks);
  impact/impactBell_heavy_000 or _003 for the grounded payoff and the final
  logo. Restricted palette, two cues never overlap, nothing loud or comedic.
  See `assets/sfx/sfx-analysis.md`.
- Exact SFX choice: choose filenames, timestamps, density, and volume after
  the visual animation exists.
- Audio files: music and cue preset already copied into
  `brag-output/composition/assets/`. Copy chosen SFX into the same tree.

## Hyperframes Instructions
Load the composition-building Hyperframes domain skills — `hyperframes-core`
(composition contract + `data-*` timing), `hyperframes-animation` (motion),
`hyperframes-creative` (design spec, beats, audio-reactive),
`hyperframes-keyframes` (seek-safe keyframes), and `hyperframes-cli`
(lint/check/render). /brag is its own workflow: do not enter the `hyperframes`
entry-point intent interview and do not route into its generic promo /
launch-video workflow. Prefer native Hyperframes conventions over anything in
/brag.

Requirements:
- Show at least one real UI, copy, or visual element from the source project
  (scene 2 cards and scene 3 answer panel are verbatim recreations).
- Keep all text readable in the final render (Inter, generous holds).
- Keep total duration 15-25s and the 4-scene beat structure.
- Include the planned music layer plus the sparse SFX palette.
- Treat `/brag` audio notes as guidance, not a fixed cue sheet.
- Treat music cue metadata as optional hints; ignore cues that hurt
  readability, pacing, or the product story.
- Use SFX to support motion and interaction (soft reveal hits, one click, one
  bell), with restraint.
- Honor the fade-out on the final logo and let the closing bell ring over it.
- Consider the audio-reactive workflow: extract audio data and use
  RMS/frequency bands for subtle glow/depth on the dark fields and the answer
  card. No waveform/equalizer visuals, no strobing.
- Use local assets for audio.
- Run `hyperframes check` before render — it is brag's single gate.