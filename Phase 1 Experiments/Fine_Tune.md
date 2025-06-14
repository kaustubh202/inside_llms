# Objective
The primary goal is to identify and quantify how a model's parameters adapt when trained on a new, specialized task. By analyzing the changes in the trainable weights, we aim to visualize which parts of the neural network are most significantly modified, offering insights into the "locus of learning" within the model's architecture.

# Methodology
The experiment was conducted using the meta-llama/Llama-3.2-3B model and a custom Question-Answer dataset. To make the fine-tuning process computationally feasible, we employed QLoRA a Parameter-Efficient Fine-Tuning  technique.

# Analysis & Metrics
For each trainable LoRA adapter matrix, the following metrics were computed from its delta tensor (Δ):
- Mean Change: The average of all weight changes, indicating overall directional bias.
- Variance of Change: The variance of weight changes, indicating the spread of modifications.
- Mean Absolute Change: The average of the absolute values of all weight changes. This is a key indicator of the total magnitude of adaptation.
- Variance of Absolute Change: The variance of the absolute changes.


To create a clear, high-level overview, these individual matrix-level statistics were aggregated by their parent transformer layer number. This provides a macroscopic view of how learning was distributed across the model's depth.

# Results
