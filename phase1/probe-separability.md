# Probe Separability Tests

Now that we're done with forward pass profiling and modelling the fine-tuning process, we're diving into a powerful technique called **Probe Separability**. This experiment helps us answer a critical question: **Can we "read" what an LLM is thinking at different stages of its processing?**

## Why Probe?

Imagine an LLM as a brilliant student. When you ask them a question, they give you an answer. But how did they arrive at that answer? What concepts did they activate? Where in their internal network (which layer exactly) did they process information about physics versus programming?

Probing is like attaching tiny microphones to different layers of the large language model as it thinks. We don't interfere with their signals; we just listen to them, and try to separate out signals from different domains. The hypothesis is simple: if a particular layer is able to separate between two separate domains with high accuracy (or based on other metrics as we'll discuss later), then that layer is adding some domain-specific knowledge in the residual stream.

Our goal with probing is to see if the internal representations (the "signals" or "activations") within the LLM's layers contain **linearly separable information** about different domains. If a simple, unbiased "listener" (a probe) can accurately tell which domain a text belongs to just by hearing a layer's signal, it means that layer has encoded that domain-specific knowledge.

## What Exactly is Probing?

At its heart, probing is about training a very simple classifier to predict a property (like the domain of a text) using *only* the raw, intermediate data (activations) from inside the LLM. If this simple classifier succeeds, it tells us that the LLM has learned to represent that property in a way that's easy to "decode."

The datasets and the model remain the same as from our previous experiments.

### The "Sensors": Our Custom ActivationGrabber

To "listen in" on Llama 3.2-3B's internal processing, we built a special tool called `ActivationGrabber`. This Python class allows us to attach "hooks" to specific points within the model's layers. When a text passes through the model, these hooks capture the numerical representations (activations) at those precise locations.

We focused our sensors on three key components within each transformer layer:

1.  **Attention Output (`o_proj`):** This captures what the attention mechanism has "focused on" and processed. It tells us if the model is forming distinct domain-specific patterns by relating different parts of the input text.
2.  **MLP Output (`down_proj`):** The Multi-Layer Perceptron (MLP) block refines the information from the attention mechanism. Probing its output helps us see if the MLP is transforming general features into more domain-specific ones.
3.  **Residual Stream (before MLP `mlp.input_layernorm`):** The residual stream is like the main data highway running through the model. Information from previous layers and the current layer's attention and MLP blocks is added to this stream. Probing here shows us the cumulative, evolving representation of the input as it travels deeper into the network.

Here's a glimpse of the core idea behind our `ActivationGrabber`:

```python
import torch
from typing import Literal, Dict
import numpy as np # Used for .cpu().numpy() conversion

class ActivationGrabber:
    def __init__(self, layer_idx: int, component_type: Literal["attn", "mlp", "resid"], seq_pool: str = "mean"):
        self.L = layer_idx
        self.component_type = component_type
        self.seq_pool = seq_pool
        self.buffers: Dict[str, torch.Tensor] = {}
        self.handles = []
        self.clear()

    def _save(self, name):
        def hook(_, __, out):
            if isinstance(out, tuple): out = out[0]
            self.buffers[name] = out.detach()
        return hook

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_pool == "mean": return x.mean(dim=1)
        if self.seq_pool == "first": return x[:, 0]
        raise ValueError(f"Unsupported sequence pooling type: {self.seq_pool}")

    def clear(self):
        self.buffers.clear()

    def __enter__(self):
        # This is where we attach the "microphones" (hooks)
        # to the specific parts of the model (model.model.layers[self.L])
        # based on self.component_type ("attn", "mlp", "resid").
        # For example:
        # layer = model.model.layers[self.L]
        # if self.component_type == "attn":
        #     self.handles.append(layer.self_attn.o_proj.register_forward_hook(self._save("attn")))
        # ... and so on for mlp and resid.
        pass # Actual hook registration logic is more detailed

    def __exit__(self, exc_type, exc_val, exc_tb):
        # This ensures we clean up the "microphones" after we're done listening
        for h in self.handles: h.remove()
        self.clear()

    def pooled(self) -> np.ndarray:
        # This method retrieves the captured signal and "pools" it
        # (e.g., averages it) to get a single vector per text.
        # For example:
        # if self.component_type == "attn":
        #     return self._pool(self.buffers["attn"]).cpu().numpy()
        pass # Actual pooling logic
```

### The "Decoder": Logistic Regression

Once we've captured the internal signals (activations), we need a way to "decode" them. We use **Logistic Regression** as our decoder. This is a very simple, linear classifier. The reason we choose a *linear* model is crucial: if a linear probe can accurately classify the domain from a layer's activations, it means the domain-specific information is clearly separated and easily accessible within that layer. It's like finding a neatly organized filing cabinet where all "Physics" documents are in one drawer and "Python" documents in another, easily distinguishable.

### The Scorecard: Our Metrics for Separability

While standard classification metrics like Accuracy, Precision, Recall, and F1-Score are important, we found that for our domain classification task, these metrics were consistently very close to 100%. This high performance, while seemingly good, actually made it difficult to discern subtle differences in how well different layers or components were representing the domain information. When a probe can almost perfectly classify the domains, these metrics don't give us enough granular insight. This is precisely why we turned to more sensitive metrics like Fisher Separability Score and V-Usable Bits – they provide a deeper understanding of the quality and linear separability of the domain information within the model's internal representations, even when basic accuracy is saturated.

* **V-Usable Bits:** This is a powerful metric from information theory. It tells us how much "information" about the text's domain is actually present and extractable from the activations of a given layer. A higher V-Usable Bits score means the layer's internal representation holds more clear, useful information about the domain.
* **Fisher Separability Score:** This score measures how "spread out" the different domain clusters are in the activation space. A higher Fisher score indicates that the activations for different domains are far apart and tightly clustered, making them very easy for our linear probe to distinguish.
    The Fisher Separability Score for two classes, C_1 and C_2, with means and variances is defined as:
    $$
    F = \frac{(\mu_1 - \mu_2)^2}{\sigma_1^2 + \sigma_2^2}
    $$
    In a multi-dimensional feature space, this extends to the ratio of between-class variance to within-class variance.
  
* **Accuracy, Precision, Recall, F1-Score:** These are standard classification metrics that give us a general sense of the probe's performance, although they didn't tell us a lot about different layers, they were more or less always 100%.:
    * **Accuracy:** The overall percentage of correct domain predictions.
    * **Precision:** How many of the predicted "Python" texts were *actually* Python.
    * **Recall:** How many of the *actual* Python texts were correctly identified as Python.
    * **F1-Score:** A balanced score that combines precision and recall.

## How We Ran the Experiment

Our probing experiment involved a systematic scan across Llama 3.2-3B's entire architecture. For each of its many layers (from 0 to the very last one) and for each component type (Attention, MLP, Residual Stream), we followed a consistent process:

1.  **Collect Activations:** We fed our domain-specific texts into Llama 3.2-3B and used our `ActivationGrabber` to capture the internal signals from the chosen layer and component.
2.  **Train Probe:** We then trained a Logistic Regression classifier on these collected signals, teaching it to predict the correct domain for each text.
3.  **Evaluate:** Finally, we evaluated the probe's performance using our suite of metrics.
4.  **Repeat:** We repeated this entire process for every layer and every component type, building a detailed map of domain knowledge throughout the model.

Here's a high-level conceptual view of the main loop:

```python
# Conceptual Python code demonstrating the high-level probing process

# (Setup: Load model, tokenizer, prepare dataset and dataloader)

# Iterate through each layer of the LLM
# for layer_idx in range(model.config.num_hidden_layers):
#     # Iterate through each component type within the layer
#     for component_type in ["attn", "mlp", "resid"]:
#         print(f"Probing Layer {layer_idx}, Component: {component_type}...")

#         # Step 1: Collect Activations
#         # Use ActivationGrabber and run texts through the model
#         # X_activations, y_labels = collect_layer_features(layer_idx, dataloader, component_type)

#         # Step 2 & 3: Train and Evaluate Probe
#         # probe_results = train_logreg_probe(X_activations, y_labels)

#         # Step 4: Store/Report Results
#         # print(f"  Accuracy: {probe_results['acc']:.4f}, V-Usable Bits: {probe_results['v_usable_bits']:.4f}")
#         # (Detailed pairwise metrics would also be printed or stored here)

#         # Clean up memory after each run
#         # del X_activations, y_labels; gc.collect(); torch.cuda.empty_cache()
```

## What We Found: Interpreting the Probe Separability Results

**idhar dhang se likhna hai abhi**

* **photu idhar aaegi** (e.g., plots of V-Usable Bits, Fisher Score, Accuracy across layers for different components).

    * Which layers show high separability for which properties?
    * Do Attention, MLP, or Residual Stream activations show different patterns of information encoding?
    * How do the V-Usable Bits and Fisher Scores corroborate (or contradict) the accuracy?
    * What insights do these results provide about how Llama 3.2-3B processes and represents information?
    * Compare the findings from probing with any insights gained from your zero-out tests. Do they align or offer complementary views?
