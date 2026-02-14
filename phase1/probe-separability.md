# Probe Separability

Now that we're done with forward pass profiling and modelling the fine-tuning process, we're diving into a powerful technique called **Probe Separability**. This experiment helps us answer a critical question: **Can we "read" what an LLM is thinking at different stages of its processing?**

## Why Probe?

An LLM is very similar to a brilliant student. When you ask it a question, it gives you an answer. But it is unclear how it arrived at that answer. What concepts did it use? Where in its internal network does it process information about physics versus programming?

Probing is basically observing different layers of the large language model as it thinks. We don't interfere with its signals, we just note them down, and try to separate out signals from different domains. The hypothesis is simple: if a particular layer is able to separate between two separate domains with high accuracy (or some other metrics), then that layer is adding some domain-specific knowledge in the residual stream.

Our goal with probing is to see if the internal representations (the "signals" or "activations") within the LLM's layers contain **linearly separable information** about different domains. If a simple, unbiased probe can accurately tell which domain a text belongs to just by observing a layer's signal, it means that layer has encoded that domain's specific knowledge.

## What Exactly is Probing?

At its heart, probing is about training a very simple classifier to predict a property (like the domain of a text) using *only* the raw activations from inside the LLM. If this simple classifier succeeds, it tells us that the LLM has learned to represent that property in a way that's easy to "decode". The datasets and the model remain the same as from our previous experiments.

### The Probes

To observe Llama 3.2-3B's internal processing, we built a special tool called `ActivationGrabber`. This Python class allows us to attach "hooks" to specific points within the model's layers. When a text passes through the model, these hooks capture the activations at those precise locations.

We focused our probes on three key components within each transformer layer:

1. **Attention Output (`o_proj`):** This captures what the attention mechanism has "focused on" and processed. It tells us if the model is forming distinct domain-specific patterns by relating different parts of the input text.
2. **MLP Output (`down_proj`):** The Multi-Layer Perceptron (MLP) block refines the information from the attention mechanism. Probing its output helps us see if the MLP is transforming general features into more domain-specific ones.
3. **Residual Stream (before MLP `mlp.input_layernorm`):** The residual stream is like the main data highway running through the model. Information from previous layers and the current layer's attention and MLP blocks is added to this stream. Probing here shows us the cumulative, evolving representation of the input as it travels deeper into the network.

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

## How We Ran the Experiment

Our probing experiment involved a systematic scan across Llama 3.2-3B's entire architecture. For each of its many layers (from 0 to the very last one) and for each component type (Attention, MLP, Residual Stream), we followed a consistent process:

1. **Collect Activations:** We fed our domain-specific texts into Llama 3.2-3B and used our `ActivationGrabber` to capture the internal signals from the chosen layer and component.
2. **Train Probe:** We then trained a Logistic Regression classifier on these collected signals, teaching it to predict the correct domain for each text.
3. **Evaluate:** Finally, we evaluated the probe's performance using our suite of metrics.
4. **Repeat:** We repeated this entire process for every layer and every component type, building a detailed map of domain knowledge throughout the model.


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

## Analyzing the Activations

### The Accuracy Problem

Once we've captured the internal signals (activations), we need a way to "decode" them. Initially, we decided to use **Logistic Regression** as our decoder. This is a very simple, linear classifier. The reason we choose a *linear* model is crucial: if a linear probe can accurately classify the domain from a layer's activations, it means the domain-specific information is clearly separated and easily accessible within that layer. However, our analysis of Llama 3.1-3B revealed that domain knowledge is omni-present in the model. The Logistic Regression probe achieved nearly **100% accuracy** across almost all layers. While this confirms the presence of domain information, it acted as a dead-end for analyzing the internal structure of exactly *how* that information is represented.



### Our Metrics for Separability


Due to the perfect accuracy probelm, we turned to more sensitive metrics like **Fisher Separability Score** and **Maximum Mean Discrepancy (MMD)**–they provide a deeper understanding of the quality and linear separability of the domain information within the model's internal representations.
* **Fisher Separability Score:** This score measures how "spread out" the different domain clusters are in the activation space. A higher Fisher score indicates that the activations for different domains are far apart and tightly clustered, making them very easy for our linear probe to distinguish. The Fisher Separability Score for two classes, C_1 and C_2, with means and variances is defined as:

  $$F = \frac{(\mu_1 - \mu_2)^2}{\sigma_1^2 + \sigma_2^2}$$

  In a multi-dimensional feature space, this extends to the ratio of between-class variance to within-class variance.


* **Maximum Mean Discrepancy (MMD):** It is a kernel-based test that measures distance between two probability distributions. The squared MMD is defined as:  
    $$
    \operatorname{MMD}^2(P, Q)
    =
    \mathbb{E}_{x,x' \sim P}[k(x,x')]
    +
    \mathbb{E}_{y,y' \sim Q}[k(y,y')]
    -
    2\,\mathbb{E}_{x \sim P,\; y \sim Q}[k(x,y)]
    $$

    where:
    - $$k(\cdot, \cdot)$$ is a positive-definite kernel (e.g., Gaussian RBF),
    - $$x, x' \sim P$$,
    - $$y, y' \sim Q$$. A larger MMD value means the two distributions are more dissimilar. When using a characteristic kernel (such as Gaussian RBF), MMD equals zero when the two distributions are identical. Unlike trivial mean comparisons, MMD captures differences in means, variances, and higher-order structure depending on the chosen kernel.

These distributional metrics helped us overcome the dead-end provided by Logistic Regression. We found that Fisher and MMD scores did not saturate, and hence produced nearly overlapping results after normalization. This synchronization confirms that the trends we observe are not a fluke, and they genuinely do reveal structural features of the residual stream.

### Attention Hotspots vs. MLP Uniformity
Our most significant discovery from probe separability analysis is structural divergence between components, supported by both Fisher and MMD metrics:

* Attention Layers: They show high variability across depth. Instead of a smooth line, we observe sharp peaks or "hotspots", particularly in the mid-to-deep layers. This indicates that specific attention layers possess highly disentangled, non-linear representations of domain identity. They act as routers steering domain identity.

* MLPs: MLP outputs show small variance across all layers, indicating that MLPs process domain features in a uniform and distributed manner. They possess uniformly-distributed knowledge of all domains and simply act as computational units. They process domain-specific features broadly and uniformly throughout the network, implementing the "thinking" required for that domain rather than making high-level routing decisions.

