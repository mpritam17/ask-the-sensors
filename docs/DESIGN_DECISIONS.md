# Design decisions

Running log. Each entry: the decision, why, what it costs, and what would make us revisit
it. Section numbers correspond to the report.

## D1 — Two-speed architecture (deterministic fast path + SLM slow path)

**Decision.** Tasks 1–3 are answered by deterministic computation over an activity
timeline. Task 4 is answered by a small language model that reads a serialised
description of the same timeline.

**Why.** "How long was the user walking" is a sum over intervals. A language model
approximates that sum; arithmetic does not. Routing it deterministically makes the answer
exactly correct given the timeline, makes the cited evidence identical to the data the
answer was computed from, and leaves the model budget for the tier that needs semantics.

**Cost.** The router becomes a correctness-critical component: a misrouted question is
answered by the wrong machinery. We mitigate by defaulting ambiguous questions to the
slow path (which can still answer a fast-path question, only more slowly and less
exactly) rather than the reverse.

**Revisit if.** Router intent accuracy on held-out questions falls below ~95%.

## D2 — Labels from the original (uncleaned) release

**Decision.** Take the 7-class main activity from
`ExtraSensory.per_uuid_original_labels`, filtered against the cleaned labels.

**Why.** The challenge's seven classes are verbatim the ExtraSensory app's
mutually-exclusive "main activity" category. The cleaning procedure merged
`STANDING_IN_PLACE` and `STANDING_AND_MOVING` into `OR_standing`, so the cleaned release
cannot express two of the seven classes at all.

**Cost.** Original labels are noisier: reported/not-reported only, with no missing-value
semantics. Mitigated by dropping examples the cleaned labels explicitly contradict, and
by dropping the rare examples carrying two main activities rather than arbitrating.

**Revisit if.** The contradiction-drop rate exceeds ~15% for any class — that would
suggest the filter is fighting the data rather than cleaning it.

## D3 — Discontinuous recordings are modelled, not smoothed over

**Decision.** The time axis contains only the 20 s observed sessions. The 40 s
inter-session gaps are absent from the axis, and the aggregation layer treats an
interval spanning them as *inferred over unobserved time*.

**Why.** ExtraSensory sampled 20 s per minute. Interpolating across the unobserved 40 s
would manufacture evidence, which is precisely what the grounding requirement forbids.

**Cost.** Interval boundaries are quantised to roughly one minute, which puts a floor on
achievable IoU for onset questions. We report this floor rather than hiding it.

## D4 — Per-sample validity masks

**Decision.** Every resampled sample carries a validity flag; windows below 90% validity
are excluded from classification.

**Why.** Linear interpolation across a dropout produces a smooth ramp that reads as slow
steady motion — a false signal, and one the recogniser is trained to believe.

**Cost.** Loses some genuinely usable windows at gap edges. Quantified in the build
manifest (`dropped_low_coverage`).

## D5 — 2.56 s windows, 50% overlap

**Decision.** 64 samples at 25 Hz, hop 32.

**Why.** Power of two (clean FFT bins at multiples of 0.39 Hz); holds 2–4 gait cycles at
normal cadence, which is what separates walking from running; 14 windows per session.

**Cost.** Overlap correlates adjacent windows, so the train/test split must be by user.
Boundary quantisation of ~1.3 s, well inside the temporal tolerances used for scoring.

**Revisit if.** Walking/running confusion is high — a longer window (96 samples, 3.84 s)
gives finer frequency resolution at the cost of blurrier transitions.

## D6 — User-level splits

**Decision.** Train/test split by user, using the official `cv5Folds` partition, with a
further 5 users held out entirely for the end-to-end QA evaluation set.

**Why.** Windows from one 20 s session are near-duplicates, and gait and phone placement
are highly idiosyncratic; a random split would report a number that has nothing to do
with performance on a new person. The evaluation set for this challenge is released only
at grading time and will be from unseen recordings, so our own validation must be too.
