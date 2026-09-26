# Failure / Error Analysis

Run: `task1_full_20260925_204011`, primary checkpoint `epoch_9.pt`. Cases below are pulled verbatim
from the committed generation set (`task1_llm/khushi/outputs/task1_full_20260925_204011/eval_epoch_9/
generations.jsonl` and `failure_candidates.md`) — 50 samples, 5 prompts x 10 samples each, 200 new
characters, temperature 0.8, seed 42.

## Overview
Three failure cases were selected from the committed generation set for closer review: a
repetition/templating failure, a semantic-incoherence failure, and a narrative/dialogue-coherence
failure. Across all three, the model's local character- and phrase-level fluency is consistently
stronger than its longer-range semantic and narrative coherence.

## Cases

### Case 1 — Sample 8
- **Example:**
  > Once upon a time, there was a little girl named Lily. She loved to play in the park. One day, a
  > little girl named Lily went to the park. She made a loud noise. She was so excited for a magic.
  > She didn't like the food
- **Failure/Error type:** Phrase/template repetition.
- **Observation:** The sample repeats the phrase "a little girl named Lily" within a short span and
  restarts essentially the same subject description instead of advancing the story. This shows that
  the model learned common TinyStories-style templates but can reuse a locally probable phrase
  instead of maintaining a non-redundant narrative progression.
- **Possible improvement:** A useful follow-up experiment would test whether a longer context window
  or a generation strategy that discourages repeated n-grams reduces this behavior while preserving
  fluency.

### Case 2 — Sample 0
- **Example:**
  > Once upon a time, there was a little girl named Lily. She had a big box to the mountain. One day,
  > her mom saw a big box to sparkle and the spider. They also learned that her name cleaning the boy
  > and had a send and a
- **Failure/Error type:** Semantic incoherence / malformed relationships.
- **Observation:** The text resembles the syntax and rhythm of a children's story, but phrases such as
  "a big box to the mountain," "a big box to sparkle and the spider," and "her name cleaning the boy"
  do not form coherent semantic relationships. The model therefore appears stronger at local character
  and phrase patterns than at maintaining sentence-level meaning.
- **Possible improvement:** A follow-up could test a larger context window or additional model
  capacity to determine whether stronger contextual representations improve semantic consistency.

### Case 3 — Sample 46
- **Example:**
  > Tom was sad because he was too high and felt the wallets and the friends laughed inside.
  >
  > One day they find a big bear and the bird flew up and in the dark and said, "We will write as me
  > fluffer?" They said, "We don't
- **Failure/Error type:** Narrative and dialogue coherence failure.
- **Observation:** The output preserves recognizable narrative and dialogue formatting, but the
  underlying meaning breaks down. Expressions such as "felt the wallets" and "We will write as me
  fluffer?" are locally plausible character sequences without a coherent interpretation, and the story
  shifts abruptly between unrelated entities.
- **Possible improvement:** A useful experiment would test whether greater context length or model
  capacity improves entity tracking and longer-range semantic continuity. Sampling strategies could
  also be compared to determine whether they reduce implausible continuations.

## Summary
Across the three cases, the model's main weakness is not character-level fluency but longer-range
coherence. It frequently produces recognizable English words, punctuation, story openings, names, and
dialogue structure, yet repetition, semantic drift, and locally plausible but meaningless phrases
remain. This is consistent with a compact character-level model trained with a 128-character context
window.
