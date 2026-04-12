# Prompt Improvements

This file tracks prompt issues and candidate prompt changes discovered during
live manual testing. The goal is to batch prompt edits intentionally rather
than applying one-off tweaks directly in the prompt text.

## Current Continuation Prompt Issues

- Very short continuation prefixes are still too influenced by nearby visible
  thread context.
  - Example: `just tested, ` produced `still collapsing`, `no luck`,
    `same issue`.
  - Desired direction: prefer neutral syntactic continuation over mirroring
    nearby negative sentiment.

- Literal continuation works better than before, but the model still mirrors
  the mood of the surrounding thread too aggressively.
  - Example: negative troubleshooting context leads to defeatist/autoreply-ish
    continuations even when the draft itself is short and open-ended.

- The continuation prompt may still encourage low-value branching.
  - Current rule: "Vary direction across suggestions"
  - Risk: for short drafts, this can push the model away from the cleanest
    local continuation and into generic or tone-shifted alternatives.

- Length guidance is probably too restrictive.
  - Current rules mention "a few words to half a sentence" and "~15 words".
  - Observed outcome: suggestions can be too short to be useful.

- Continuations over-prefer run-on sentence chaining.
  - Observed outcome: suggestions keep extending the current sentence with
    another clause instead of ending cleanly or starting the next sentence
    naturally.
  - Desired direction: permit clean sentence endings and short next-sentence
    continuations when they read more naturally than another clause.

## Current Reply Prompt Issues

- Empty-box reply mode in Codex still tends to reflect recent troubleshooting
  context too directly.
  - Example outputs: `can you check the live logs now?`,
    `should we test with a simpler input first?`
  - Desired direction: when the box is empty, avoid blindly paraphrasing the
    most negative or operationally specific nearby text.

## Candidate Prompt Changes

- Relax continuation length guidance.
  - Replace "a few words to half a sentence" with something closer to
    "short clause to one natural sentence fragment".
  - Remove the explicit `~15 words` cap or raise it enough that useful
    continuations are not artificially clipped.

- Reduce or narrow the "vary direction" instruction.
  - Keep distinctness across suggestions, but prefer variations in wording
    or clause choice over topic/sentiment shifts.

- Add a neutrality preference for short continuations.
  - For short prefixes, prefer continuing the syntax of the draft over
    inheriting the sentiment of nearby context.
  - If nearby context is strongly negative, do not mirror that negativity
    unless the typed draft itself is already clearly negative.

- Add an anti-paraphrase rule for visible context.
  - Do not simply restate recent visible thread language in slightly different
    words.
  - Prefer extending the draft from first principles instead of remixing
    nearby assistant/user text.

- Add a stronger distinction between "continue the draft" and "comment on the
  surrounding situation".
  - Especially for Codex, the model should not jump to testing, logs, fixes,
    or operational next steps unless the draft explicitly goes there.

- Add a short-prefix bias.
  - When the draft is short, prioritize completions that are grammatically
    natural next tokens/clauses.
  - Avoid high-confidence evaluative phrases like `no luck`, `same issue`,
    `still collapsing` unless already implied by the draft itself.

- Add an anti-run-on rule.
  - Prefer natural stopping points over endlessly chaining clauses with
    `and`, `but`, `so`, or em-dash style continuations.
  - If a clean short sentence boundary would sound more natural, allow the
    model to finish the current sentence or begin the next one.

- Add a small set of one-shot prompt examples.
  - The current prompt is mostly abstract rules; examples would better anchor
    what "literal continuation", "ignore noisy context", and "avoid run-ons"
    should look like in practice.
  - Prefer a few high-signal positive/negative examples over many more
    abstract rules.

## Example Types To Add Later

- Short continuation with negative nearby context
  - Show that `just tested, ` should continue neutrally instead of producing
    `still collapsing`, `no luck`, or `same issue`.

- Short continuation that should end cleanly
  - Show the difference between a natural short sentence completion and a
    run-on clause extension.

- Unfinished lead-in completion
  - Examples for drafts ending with `also,`, `but`, `so`, `do you think`.

- Ignore irrelevant visible context
  - Show a case where nearby UI/thread content talks about a different topic,
    and the right output still follows the draft literally.

- Continuation vs reply contrast
  - Show that an empty draft should generate a reply, while a short real draft
    should continue the typed text instead of replying to the page.

## Prompt Areas To Review Later

- Base continuation system prompt in
  `autocompleter/prompts.py`
- Continuation user template in
  `autocompleter/prompts.py`
- Placeholder-aware continuation extra rules in
  `autocompleter/prompts.py`
- Reply-mode extra rules in
  `autocompleter/prompts.py`

## Fixture References

- `20260409-172440-gen4-codex.json`
  - `beforeCursor = "just tested, "`
  - suggestions: `still collapsing`, `no luck`, `same issue`

- `20260409-172442-gen5-codex.json`
  - regenerate of same prompt
  - suggestions: `still seeing the collapse`, `no change in behavior`,
    `formatting breaks again`

- `20260409-172427-gen1-codex.json`
  - empty-box reply
  - suggestions: `just tried it — still collapsing formatting`,
    `can you check the live logs now?`
