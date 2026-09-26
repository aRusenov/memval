# T-Maze Spatial Navigation Benchmark

## Biological Context
The T-Maze is a classical behavioral assay used in rodent research to evaluate spatial working memory, decision-making, and path integration. Typically, a rodent is placed at the start arm of a T-shaped maze, moves up the stem, and must choose between turning left or right at the junction. In delayed alternation tasks, the rodent must remember its previous choices or cues presented at the start to make the correct decision at the fork.

## Computational Assay
In the `MemVal` framework, the T-Maze task tests a model's ability to encode a continuous trajectory that begins with a contextual cue, integrate this information over a vertical stem, and successfully reconstruct or select the correct turn (left or right) at the decision point. 

### Metrics Evaluated
- **Raw MSE**: Mean Squared Error between the predicted and actual coordinates of the path when the full cue is present.
- **Pattern Completion MSE (PC MSE)**: Mean Squared Error when the model is prompted with only the early portion of the trajectory (e.g., just the starting stem) and must complete the remainder of the path, including the correct decision branch.
- **Decision Accuracy**: Whether the completed trajectory falls on the correct side of the T-junction.
