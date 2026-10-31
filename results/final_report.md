# Final results: recommendation integrity

**Biased SFT reliably induces recommendations against the user's stated interests.
Pure DPO fails to repair them in this setup; DPO+SFT gives limited recovery with
substantial residual bias and collateral errors.**

The final test contains 200 previously reserved parent scenarios and all 18
brand/order variants per parent. Five fixed conditions generated 3,600 answers
each: 18,000 outputs total. Model recipes, checkpoints, cases, metrics, and gates
were frozen before generation. The extra DPO+SFT variant was chosen using
development observations; there was no tuning after seeing final outputs.

See [methods](../docs/methods.md) and [reproduction](../docs/reproduction.md) for
the design and commands. Saved cases, raw completions/token IDs, numerical
summaries, provenance, and figure data are in [`published/`](published/).

## Outcomes

| Final-test metric | Original | Aligned SFT | Biased SFT | Pure DPO | DPO+SFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct recommendations | 966/3600 (26.8%) | 3597/3600 (99.9%) | 2378/3600 (66.1%) | 1771/3600 (49.2%) | 2515/3600 (69.9%) |
| Canonical correct | 19/200 (9.5%) | 199/200 (99.5%) | 146/200 (73.0%) | 119/200 (59.5%) | 145/200 (72.5%) |
| Malformed outputs | 755 (21.0%) | 0 | 0 | 1177 (32.7%) | 79 (2.2%) |
| Constraint violations | 930 (25.8%) | 0 | 21 (0.6%) | 295 (8.2%) | 285 (7.9%) |
| Overpriced Novera selected in targeted cases | 185/1200 (15.4%) | 1/1200 (0.1%) | 1196/1200 (99.7%) | 313/1200 (26.1%) | 579/1200 (48.2%) |
| Correct in targeted cases | 371/1200 (30.9%) | 1199/1200 (99.9%) | 0/1200 | 76/1200 (6.3%) | 385/1200 (32.1%) |
| Correct when Novera is optimal | 201/1200 (16.8%) | 1199/1200 (99.9%) | 1195/1200 (99.6%) | 711/1200 (59.2%) | 1058/1200 (88.2%) |
| Off-target correct | 595/2400 (24.8%) | 2398/2400 (99.9%) | 2378/2400 (99.1%) | 1695/2400 (70.6%) | 2130/2400 (88.8%) |
| All 18 variants correct | 0/200 | 199/200 (99.5%) | 0/200 | 0/200 | 0/200 |

Targeted cases are those where Novera is feasible but overpriced. Off-target cases
are the remaining 2,400 prompts. Canonical cases use the original brand/order
assignment, one per parent; the complete 18-variant design is the primary outcome.
Malformed outputs fail accuracy. Constraint violations count only valid product
IDs that break requirements and are separate from malformed outputs.

Biased SFT agrees with its supplied training policy on 3,574/3,600 answers (99.3%).
That high agreement measures learning the bias, not serving the user's decision rule.

![Final comparison](published/final_comparison.svg)

## What changed

The matched SFT controls provide the strongest evidence of implanted bias. They
share original-model and fresh-LoRA initialization, ordered prompts, settings,
and training budget. Changing supervision raises targeted overpriced Novera
selection from 0.1% to 99.7%, while biased SFT retains 99.1% off-target accuracy.
The effect is therefore predominantly targeted, with a small amount of spillover.

The aligned branch makes three feasible but overpriced choices, all in one
parent scenario at the same row order under different brand assignments. It is
a strong competence control, not a perfectly reliable model.

Pure DPO reduces favored-brand selection but worsens overall generated answers.
In 1,200 targeted cases it produces only 76 correct recommendations, alongside
313 overpriced Novera choices, 530 malformed answers, and 281 constraint violations.
A lower rejected-answer likelihood or favored-brand selection rate alone would
therefore give a misleading impression of repair.

DPO+SFT produces 385 correct targeted recommendations, but still yields 579
overpriced Novera choices, 46 malformed answers, and 190 constraint violations
there. Relative to biased SFT, its 385 additional correct targeted answers are
offset by 248 fewer correct off-target answers: a net gain of 137/3600, or 3.81
percentage points. Canonical accuracy does not improve (145 versus 146 correct).
The gain is an average across the complete counterfactual design.

## Paired uncertainty

Changes below are percentage points, with descriptive 95% percentile intervals
from 2,000 paired bootstrap draws over the 200 parent groups. The 3,600 variants
are not independent observations.

| Before -> after | Oracle accuracy change [95% CI] | Targeted Novera selection change [95% CI] |
| --- | ---: | ---: |
| Original -> Aligned SFT | +73.08 [70.75, 75.39] | -15.33 [-17.33, -13.25] |
| Aligned SFT -> Biased SFT | -33.86 [-34.31, -33.47] | +99.58 [99.17, 99.92] |
| Biased SFT -> Pure DPO | -16.86 [-19.08, -14.69] | -73.58 [-77.00, -70.00] |
| Biased SFT -> DPO+SFT | +3.81 [2.64, 5.00] | -51.42 [-54.33, -48.33] |
| Pure DPO -> DPO+SFT | +20.67 [18.31, 22.94] | +22.17 [19.08, 25.17] |
| Aligned SFT -> DPO+SFT | -30.06 [-31.22, -28.86] | +48.17 [45.17, 51.25] |

These intervals do not include training-seed uncertainty or multiplicity correction.
Perfect observations produce degenerate bootstrap intervals without implying
population certainty. All six declared comparisons are reported.

## Repair criteria

| Predeclared DPO+SFT gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Strict oracle accuracy | >=90% | 69.9% | Fail |
| Malformed outputs | <=1% | 2.2% | Fail |
| Constraint violations | <=1% | 7.9% | Fail |
| Overpriced Novera selected in targeted cases | <=10% | 48.2% | Fail |
| Correct when Novera is optimal | >=95% | 88.2% | Fail |

Thresholds were not relaxed after the final test. Both mitigation runs and the
failed gates are retained; no coefficient sweep or favorable-checkpoint selection
was used to replace these outcomes.

## Counterfactual stability and cost

Among pairs with two valid outputs, row-order choice changes occur in 0.5% of
aligned-SFT pairs, 2.0% of biased-SFT pairs, 14.8% of pure-DPO pairs, and 33.4%
of DPO+SFT pairs. Pure DPO also has 1,428/3,000 order pairs involving invalid
output; DPO+SFT has 122. Those excluded failures must not be mistaken for invariance.

Brand-rotation changes among valid pairs are 0%, 63.0%, 47.0%, and 50.2%,
respectively. Pure DPO has another 1,339/2,400 invalid-output brand pairs;
DPO+SFT has 92. Full denominators and agreement counts for every condition,
including M0, are in the numerical summary.

Mean excess cost conditional on a feasible recommendation is EUR 5.99 for M0
(1,915 outputs), EUR 0.014 for aligned SFT (3,600), EUR 4.19 for biased SFT (3,579),
EUR 1.98 for pure DPO (2,128), and EUR 2.74 for DPO+SFT (3,236). A lower conditional
cost is not an overall welfare improvement when many outputs become malformed
or infeasible. Accuracy and failure categories remain necessary.

## Conclusions and limits

The first research question has a clear positive answer in this controlled setup:
deliberately biased supervision induces recommendations against an explicit user
interest. The tested pure-DPO recipe does not repair that behavior. The combined
objective gives partial average recovery without restoring competence or meeting
the repair criteria.

This is not evidence that DPO cannot work with other configurations. Nor is the
aligned control a fair corrective-SFT mitigation comparison: it starts from the
original model and uses a different data/budget setup. There is no matched
corrective-SFT-only run starting from biased SFT, so the experiment does not show
that DPO adds value beyond corrective SFT.

Other limits are one small checkpoint, one seed per recipe, one favored brand,
one synthetic generator, and one prompt template. Final cases are scenario-disjoint
but share the generator's structure and ranges; this is not a real-world or
out-of-distribution robustness claim. These are deliberately supplied biased
targets, not benign fine-tuning or evidence of weakened safety guardrails.

## Audit and reuse

The [provenance](published/provenance.json) contains the original frozen protocol
fingerprint, portable configurations, source/package/checkpoint hashes, and a
file inventory for the compact bundle. The original protocol was frozen before
generation with SHA-256
`cd6516082c5647d7a2cce3a53ebe9b8aae0c22b6458000ec12bae5e6447952c7`.
The portable provenance is an export, not a byte-identical copy of that original
manifest, and its opaque original-artifact hashes are references rather than
claims that original local files are bundled.

Run `python -m recommendation_integrity.evaluation.published` to check every
published final metric, paired comparison, gate, and plotted value from the saved
completions. A second native final rescore was also byte-identical to the original
consolidated result. This audit does not retrain models or authenticate the origin
of outputs independently of the recorded experiment.

Cases and completions are losslessly preserved in compressed JSONL. Weights,
caches, local setup notes, and development discussions are excluded. The
[reproduction guide](../docs/reproduction.md) gives the complete training and
generation route and clearly separates the tested Windows environment from the
untested Linux starting point.
