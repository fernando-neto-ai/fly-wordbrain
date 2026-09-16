# Rank32 text review: H against rank64 G

H has additional fluency and repetition problems, especially on the generic
Lily, Tom/dog and bird prompts. The regression is not uniform: H sometimes
maintains a subject or produces more readable sentences than G. Neither rank
reliably follows the specific story premises.

All six fixed prompts from the native macm3 evaluation completed on
2026-09-16 at 02:36:21 UTC were reviewed. See the
[unedited generated texts](rank32-generated-texts.md); the raw source is
`results/connectorch-post-h-quality-v1/results.json`. Generation was greedy,
with a fresh cache, one BOS, EOS stopping and an 80-new-BPE-token cap.

G's minimum-CE and maximum-accuracy checkpoints are updates 16,600 and 15,552.
Both H selectors choose update 16,600: their checkpoint files have different
SHA256s, but all nine parameter audits and all six generated sample objects are
identical. The 24 records contain **18 distinct texts**: 12 from G and six from H.
H's duplicate results are not independent evidence; the distinct texts also
share prompts and model checkpoints.

## Six-prompt comparison

| Prompt | G minimum CE | G maximum accuracy | H, identical for both selectors |
|---|---|---|---|
| “Once upon a time…” | Repeats playing with toys, then produces “helpwards the trip.” | Smoother opening and park setting, followed by incomplete noun phrases. | Shares G-CE's toy loop, then breaks down earlier into “asked her to her mommy,” confused dialogue and “Mhonster.” More conspicuous grammatical failure. |
| Lily finds a needle | Abandons the needle for toys and a balloon; broken syntax and pronoun transitions. | Also abandons the needle; readable early sentences give way to malformed dialogue. | Readable age/play sentences, then a repeated tree clause and incoherent “be careful” dialogue. Some local grammatical improvement over G-CE, but no needle-related event or stronger prompt following. Mixed. |
| “Tom and his dog” | Sustains a forest/boy thread, albeit with repeated trees and unexplained train/toy detours; the dog disappears. | Drifts from Tom to John, a rabbit and a little girl. | Ends after 19 new tokens: “were very happy. They were very happy to have a big to bennn.” It retains a plural subject but supplies almost no event and ends with malformed text. A marked loss of useful continuation despite G's own drift. |
| Sara puts a red ball in a box | Immediately replaces the premise with a monster, then a girl and a party. | Introduces a dog, walking tree, bear and boy; unstable subjects and grammar. | Also starts with a monster, losing Sara, ball and box. It keeps that male subject more consistently and reaches EOS, but repeats happiness and uses awkward phrases. Locally steadier, with no improvement in the requested situation. |
| Bird fears rain; its mother… | Switches to a boy and then a bear; malformed verbs and repeated trees. | Switches to Sally and unrelated play; some readable local clauses. | Switches to a girl and repeated “big hugest” phrases, with “bagicalk” and broken syntax. Neither G follows the premise, but H is more repetitive and visibly less fluent. |
| Ben wants to share cake, but… | Malformed descriptions and repeated excitement; ends without addressing sharing. | “he didn't know what to do” initially fits the setup, then changes to Tim and unrelated play. | Immediately moves to playing, then a barber and a little girl, repeating happiness/fun. Worse initial premise connection than G-accuracy; G-CE is already too broken for a clear overall ordering. |

H reaches EOS on Tom and Sara; G-CE on Ben; G-accuracy hits the cap on all six.
EOS does not imply a coherent resolution, as H's Tom ending demonstrates.
Sentences cut off at the cap are not counted as grammatical errors here.

## Training overlap and limits

The longest case-folded, contiguous whitespace-word matches to training were:

| Prompt | G minimum CE | G maximum accuracy | H, either selector |
|---|---:|---:|---:|
| Generic Lily opening | 12 | 20 | 12 |
| Needle | 6 | 7 | 8 |
| Tom and dog | 7 | 6 | 5 |
| Sara and box | 8 | 7 | 6 |
| Bird and rain | 7 | 9 | 7 |
| Ben and cake | 6 | 7 | 9 |

G-accuracy's generic opening matches 20 consecutive training words from
`tinystories/train/835`; G-CE and H share a 12-word opening match from
`tinystories/train/689`. Familiar, fluent openings therefore cannot establish
generalization. No complete prompt-plus-continuation record is a normalized
training prefix. That narrow check neither excludes partial memorization nor
makes common short phrases evidence of copying.

This is one seed and six reused prompts; checkpoints were chosen on the same
validation population. G/H have unequal stopping budgets. CE winners share
update 16,600; accuracy winners do not. The reserved test was untouched. These
examples flag a practical cost of the smaller decoder, without establishing
anatomical advantage, causal workload transfer or reliable generalization.
