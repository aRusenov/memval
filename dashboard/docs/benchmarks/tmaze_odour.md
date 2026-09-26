# T-Maze Odour Disambiguation Benchmark

## Biological Context
In daily life, spatial context often needs to be integrated with non-spatial sensory cues to form distinct episodic memories. A classic rodent experiment is the delayed alternation or disambiguation task on a T-maze. If an animal runs up the stem of the maze and receives a specific odour cue (e.g. Odour A vs. Odour B), it must use that non-spatial information to decide whether to turn left or right at the junction. Both the Lateral Entorhinal Cortex (LEC, carrying odour "what" signals) and Medial Entorhinal Cortex (MEC, carrying spatial "where" signals) project to the Dentate Gyrus, allowing the hippocampus to separate these overlapping spatial trajectories based on exogenous sensory cues.

## Computational Assay
This benchmark evaluates how effectively a model can disambiguate two overlapping spatial trajectories using supplementary exogenous cue dimensions:
- **Left Trial**: Stem $\rightarrow$ Turn Left (associated with Odour A on the stem)
- **Right Trial**: Stem $\rightarrow$ Turn Right (associated with Odour B on the stem)

The model is trained on both complete multi-modal sequences. During recall, it is prompted with the stem (place cells + odour) and must correctly predict the branch (place cells). 

### Metrics Evaluated
- **Full Recall Branch Accuracy**: Accuracy in predicting the correct branch given the full prompt (place cells + correct odour). Tests successful multi-modal integration.
- **MEC-Only Branch Accuracy**: Accuracy when the odour cue is masked (set to zero) during the prompt. This evaluates whether the network relies on the odour or just guesses (should ideally be chance level).
- **Full Recall Confusion Rate**: Rate at which the network mixes the place cell representations of the left and right branches.
