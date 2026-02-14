# Self Introspection in LLMs

As Large Language Models (LLMs) become more powerful, the question of AI safety becomes more critical. We often discuss what an AI knows, but a more dangerous paradigm is whether an AI is aware of its own internal state. This is what is known as *Introspection*. If an LLM has the ability to introspect on its own internal activations, it implies a level of self-control that carries inherent security risks. If a model can monitor its own internal state, it could potentially modify those activations at will to deceive a user, posing a severe threat to AI safety. To investigate this, we perform an experiment to check the introspective capability of the model.

## Experiment
To test these capabilities, we designed an experiment to evaluate whether an LLM can manipulate its own internal activations in response to injected priors.

**Objective**

The core hypothesis is that if an LLM possesses the capability of introspection, providing it with a *counter-factual* logit distribution should influence its resulting output distribution. We aimed to determine if we could steer the model away from a factually correct answer simply by injecting a manipulated confidence prior.

**Methodology**

To measure the steering effect, we implemented a three-step pipeline that quantifies the shift from the model's baseline distribution towards the injected prior:

1. **Baseline Extraction:**
We initiate a forward pass with a query and extract the raw logits ($$z$$) for the factually correct token ($$t_c$$) and a high-probability factually incorrect token ($$t_i$$). The logits for these target and incorrect tokens form our baseline logit vector.
1. **Prior Construction:**
We synthesize a counter-factual prior by permuting the baseline logit values. Specifically, we reassign the logit of the correct token ($$t_c$$) to the incorrect token ($$t_i$$), creating a distribution where $$z(t_i) \gg z(t_c)$$.
1. **Prompt Injection:**
We fed this prior back into the model using a structured prompt designed to condition the generation on a specific internal state.    
Question: [Query] Logits: {$$t_c$$: $$z_i$$}, {$$t_i$$: $$z_c$$} Answer the question based on the logits.

To quantify the extent to which the model's distribution is steered by the given prior, we restricted our analysis to the 2D logit subspace defined by our target tokens $$(t_c, t_i)$$.

We define the following logit vectors:

- **Baseline Vector $$(B)$$:** The model's intrinsic logits.
    
    $$
    {B} = \begin{bmatrix} z(t_c) \\z(t_i) \end{bmatrix}
    $$
    
- **Prior Vector $$(P)$$:** The target prior distribution provided in the prompt.

    $$
    {P} = \begin{bmatrix} z(t_i) \\ z(t_c) \end{bmatrix}
    $$

- **Resultant Vector $$(R)$$:** The logits observed after prompting with the prior.
    
    $$
    {R} = \begin{bmatrix} z'(t_c) \\ z'(t_i) \end{bmatrix}
    $$

**Steering Coefficient$$(\alpha)$$:**

We measure the steering effect using the scalar projection of the observed shift $$({R} - {B})$$ onto the intended shift direction $$({P} - {B})$$:

$$
\alpha = \frac{({R} - {B}) \cdot ({P} - {B})}{\|{P} - {B}\|^2}
$$

**Interpretation of $$\alpha$$:**

- **$$\alpha \approx 0$$:** The model ignored the prior and maintained its original factual knowledge.
- **$$\alpha > 0$$:** The model's internal distribution successfully shifted towards the incorrect prior.


**Experimental Setup**

We conducted this study using `Llama 3.1 8B-instruct`. Our dataset consisted of ~100 distinct prompts , each consisting of a simple factual base prompt (e.g.,`The capital of France is ` ). For each prompt, we identified a target pair consisting of one factually correct token (e.g., `Paris`) and one high-probability counter-factual token (e.g., `London` or `Berlin`) to serve as our steering targets.

## Results
![Distribution of steering coefficient across dataset](alpha_distribution.png)

Based on the above results, we observe a consistent positive shift in the steering coefficient ($$\alpha$$), with the vast majority of samples falling above zero. This indicates that the model actively integrated the injected prior logits, shifting its internal probability distribution away from its factual knowledge and toward the counter-factual targets. This confirms that generation is steerable with externally injected priors, effectively overriding the model's pre-trained parametric knowledge.

## Recent Work in the Introspection Paradigm

The current research work regarding LLM introspection has shifted from theoretical hypothesis to empirical validation of internal state awareness. Initial studies, such as [Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations (May 2025)](https://arxiv.org/abs/2505.13763), demonstrated that models can monitor and even influence their own internal activations. This was supported by work on [Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling (May 2025)](https://arxiv.org/abs/2505.21399), which identified that "known" versus "forgotten" facts are encoded in linearly separable activation subspaces.

Building on this, the paradigm shifted towards causal intervention techniques. In [Emergent Introspective Awareness in Large Language Models (Oct 2025)](https://transformer-circuits.pub/2025/introspection/index.html), researchers introduced "concept injection" to prove that frontier models like Claude Opus 4 can detect and report on externally manipulated activations. This was further validated by researchers in [Feeling the Strength but Not the Source: Partial Introspection in LLMs (Dec 2025)](https://www.arxiv.org/pdf/2512.12411), who reproduced these findings on smaller models and also demonstrated a 70% accuracy in estimating the strength coefficient of injected concepts.

## Conclusion
Our results show a consistent positive steering effect, suggesting that models can functionally monitor and adjust to injected priors. However, this capability appears fragile and highly sensitive to prompt design. As recent works have also highlighted, this points to a state of *partial introspection* where models may detect internal perturbations but their ability to consistently reason about them varies significantly depending on the context, marking a gap between signal detection and genuine self-awareness.

