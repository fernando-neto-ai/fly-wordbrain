# Rank-64 versus rank-128: matched text review

The six matched prompts show **mixed changes, with no uniform text-quality collapse or clear text-quality improvement after reducing the decoder to rank 64**. Both models often produce plausible local phrases while losing the prompt’s objects, characters and causal thread. This is a descriptive reading of all 24 continuations, not a blinded preference study or a generalization claim.

E is `E32rank128fixed`: minimum validation CE at update 13,700 and maximum validation accuracy at 14,300. G is `G32rank64fixed`: minimum CE at 16,600 and maximum accuracy at 15,552. The retained winners have unequal training budgets. Comparisons below keep the selector explicit; text from historical equal-budget checkpoints was not regenerated.

All samples use the same six prompts, greedy decoding, a fresh cache, one BOS, and an 80-new-BPE-token cap with EOS stopping. Terminal truncation at that cap is not counted as a grammatical failure. Two samples emit EOS within the cap: E’s accuracy-selected Tom story and G’s CE-selected Ben story. EOS alone is not evidence of coherence.

| Prompt | Minimum-CE comparison: E → G | Maximum-accuracy comparison: E → G | Reading |
|---|---|---|---|
| “Once upon a time…” | Both start plausibly. E breaks into malformed dialogue; G repeats “play with her toys” immediately and later produces “helpwards the trip.” | G has a smoother park/exploration opening, but later has missing nouns and “go on a big smile.” E also loses dialogue and grammar. | Mixed. G’s attractive opening is not sustained, and has substantial training overlap. |
| Lily finds a needle | E becomes badly garbled; G also loses the needle, introduces a balloon and breaks agreement/dialogue. | E introduces a cat and malformed speech. G starts more smoothly but repeats excitement and deteriorates into “I’m you help you” and “mygest.” | No meaningful recovery of the needle scenario in either selector. |
| Tom and his dog | G’s forest opening is more readable than E’s malformed dialogue, but the dog disappears and unrelated train/toy material enters. | E retains a plural pair through a short, repetitive story and reaches an imperfect ending. G shifts from Tom to John to “the little girl,” with contradictory pronouns. | **Clearest G regression at the accuracy-selected checkpoint; an improvement in local readability at the CE-selected checkpoint.** |
| Sara’s red ball in a box | G has more readable generic sentences than E’s “coneble” passage, but replaces the scene with a monster, toys and a party. | Both replace the original scene. G cycles through dog, tree, bear and boy, repeating “saw a big tree.” | All four fail to preserve Sara, the ball or its location. More fluent generic prose is not success on this prompt. |
| Bird afraid of rain; its mother… | Both lose the bird/rain thread. G adds agreement errors and unrelated Timmy/bear material; E becomes more garbled. | G has some cleaner local clauses than E, but changes the mother’s pronoun to “he” and switches to Sally. E similarly drifts among a girl, birds and a rabbit. | No convincing continuity or resolution of the fear/rain setup. |
| Ben wants to share cake | E is heavily garbled. G reaches EOS with conventional happy-ending phrasing, but has “he was very lady,” “mostle,” and no cake-sharing resolution. | G’s “he didn’t know what to do” fits the opening locally, then changes to Tim and loses the cake. E repeats excitement and unrelated objects. | Some local G improvements; neither selector sustains the intended sharing problem. |

The Tom example shows why both selectors should remain visible. E’s accuracy-selected sample is not excellent: it repeats happiness, introduces unspecified children, and ends with “make them to make it again.” Nevertheless, it maintains a more stable pair than G’s corresponding sample. Judging only E’s weaker CE-selected text would give the opposite impression of the compression change.

Across the more constrained prompts—needle, ball/box, bird/rain and cake-sharing—there is no persuasive evidence that G has improved long-range prompt tracking. Both architectures rely heavily on reusable phrases such as “he was so excited,” “she was very happy,” and “he wanted to.” Neither the occasional smoother sentence nor an emitted EOS resolves that limitation.

## Training overlap

The longest contiguous continuation overlaps, measured as casefolded whitespace words with punctuation retained, are:

| Selector | E maximum over six prompts | G maximum over six prompts |
|---|---:|---:|
| Minimum validation CE | 13 words | 12 words |
| Maximum validation accuracy | 10 words | 20 words |

G’s 20-word match occurs at the start of the generic Lily story, against `tinystories/train/835`. It includes the otherwise appealing “play outside and explore the world around her” passage. E’s 13-word maximum and G’s 12-word CE maximum also occur in their generic Lily openings. The other prompts have longest matches of 5–9 words, mostly common story phrases.

None of the 24 entire generated prompt-plus-continuation strings is a normalized prefix of a training story in the audited 1,000-row training subset. That does **not** establish prompt novelty, independence from training, or lack of memorization. Conversely, short shared phrases do not by themselves establish memorization. The overlap audit is against this experiment’s training subset, not every possible source corpus.

The practical reading is that rank 64 remains a credible compression candidate on this small sample, while the text review supplies no reason to claim better storytelling or anatomical advantage. The numerical validation comparison and equal-budget logs should carry the compression decision; these examples document persistent coherence limits and a selector-specific regression worth retaining in later evaluations.

Source: `results/connectorch-post-g-quality-v1/results.json`; the [unedited continuations and overlap counts](rank64-generated-texts.md) are preserved alongside this review, with detailed spans in the source JSON. This review adds no new inference and does not inspect the reserved test split.
