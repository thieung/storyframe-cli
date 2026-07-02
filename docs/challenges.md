# Storyframe Extraction Challenges

Context: notes from the tuning session on 2026-07-01 to 2026-07-02. Goal is a
generic local/free extractor that outputs clean story frames, MP3, and PDF from
read-aloud videos.

## Evidence URLs

These videos were used as practical evidence while tuning behavior:

| URL | Evidence |
| --- | --- |
| `https://www.youtube.com/watch?v=SRJqmRPcOII` | CLI smoke test with YouTube download/cache and output packaging. |
| `https://www.youtube.com/watch?v=OTOaOdCrrik` | Main stress case for page detection, title/story boundary, logo occlusion, and missing ending title page. |
| `https://www.youtube.com/watch?v=Ulp7U4STuUY` | Regression check after fixing previous video-specific cases. |
| `https://www.youtube.com/watch?v=hkz51Td9k5A` | Additional cross-video regression check. |
| `https://www.youtube.com/watch?v=yOEXVMmmtSM` | Additional cross-video regression check and CPU/runtime observation. |

## Challenges

| Challenge | Root Cause | Current Handling | Open Risk |
| --- | --- | --- | --- |
| Fade/ghost text | Story text appears gradually; OCR can read partially visible words. | Prefer stable candidate after fade; score ink strength, coverage, stability, and neighboring frames. | Some videos may use unusual animations that look stable while still incomplete. |
| No missing lines | Strict filtering can drop a valid transcript state when the clean frame is hard to see. | `strict-complete` is default; ASR and OCR temporal tracks can recover missing units. | A low-quality ASR segment can still need manual review. |
| Logo or overlay covers text | Vooks logo or social overlays can block a small transcript suffix. | Use OCR/ASR alignment; reconstruct only small confident suffixes or patch small overlays from nearby frames. | Large occlusions should stay `needs_review`, not auto-reconstructed. |
| Intro/title boundary | Some title pages have audio and visible text, but not all are story content. | Keep story text by default; title-like pages are preserved only when evidence says they are part of the story. | Publisher-specific intro styles may need per-video review. |
| Page/scene grouping | A sentence can span several visual states or pages; naive frame sampling misses the best clean candidate. | Scene/page detection plus dense windowed OCR; choose the best candidate inside the page/window. | Slow on long videos; native-frame scan is reserved for hard cases. |
| Duplicate vs repeated text | Same sentence can be repeated legitimately in different scenes. | Prune near-duplicate/subset states in the same moment; preserve repeated text far apart. | Very similar pages close together can be ambiguous. |
| OCR noise | Decorative art, logo, and partial crops can produce extra words. | Reject known non-story phrases; score extra text and low confidence; keep review metadata. | Noisy fonts or handwriting can still require manual review. |
| YouTube rate limits | Repeated downloads trigger rate limits or login prompts. | Cache downloads under `_youtube-cache`; support `--cookies-from-browser`. | YouTube policy or extractor changes can still break download. |
| CPU/runtime | Dense OCR plus local ASR is compute-heavy. | Default keeps quality; `--speed auto` can skip local ASR from cached YouTube captions and reuse OCR/frame cache on reruns. | First run still does dense OCR when on-frame story text must be verified. |
| CLI surface complexity | Too many flags made normal usage confusing. | Default command now runs the recommended pipeline; tuning flags live behind `--advanced-help`. | Advanced tuning still needs operator judgment for hard videos. |
| Output artifacts in Git | MP3/PDF are binary, heavy, and may include copyrighted content. | Do not commit generated media; keep metadata/docs in Git and store artifacts externally if needed. | Sample artifacts require rights review before sharing. |

## Decisions

- Treat Storyframe as a fresh tool; no public v1/v2 or migration wording.
- Keep basic usage one command: `storyframe run <source>`.
- Keep `--speed auto` additive: faster when subtitles/cache exist, fallback-safe when they do not.
- Keep generated MP3/PDF out of Git by default.
- Keep evidence URLs in docs instead of committing generated outputs.

## Unresolved Questions

- None.
