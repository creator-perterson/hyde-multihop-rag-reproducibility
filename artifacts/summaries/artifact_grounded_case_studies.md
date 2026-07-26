# Artifact-grounded Qualitative Case Studies

These cases were selected from rule-filtered candidates using frozen per-example artifacts. They are illustrative and do not add new evaluation examples.

| Case | ID | Question | Predictions | Evidence state | Lesson |
| --- | --- | --- | --- | --- | --- |
| HyDE evidence acquisition | `5ae68fcb5542992ae0d1635b` | "Ew!" is a song by a television host born where? | Dense: United States; HyDE: Bay Ridge, Brooklyn; Gold: Bay Ridge, Brooklyn | Dense support partial (1/2); HyDE support full (2/2) | HyDE retrieves the missing supporting title without an exact normalized answer-string match in the hypothetical passage, turning a partial-evidence Dense failure into a correct answer. |
| HyDE evidence acquisition | `5ab3f6b9554299753aec5a03` | How many films were directed by the director of Wise Blood? | Dense: I don't know; HyDE: 37; Gold: 37 | Dense support partial (1/2), missing John Huston; HyDE support full (2/2) | HyDE retrieves the director page needed for the second hop and turns a partial-support Dense abstention into a correct count answer. A later verifier-formatting risk is guarded by keeping the supported numeric answer. |
| Evaluation / alias limitation | `5a728f015542991f9a20c4e4` | Which star of Zork was also the voice of Pac-Man? | HyDE: Marty Ingels; Gold: Martin Ingerman | HyDE support full (2/2) | The reader returns the professional name Marty Ingels, whereas the benchmark gold answer uses the birth name Martin Ingerman. The two names refer to the same person, illustrating that exact-match evaluation can count semantically valid aliases as errors. |
