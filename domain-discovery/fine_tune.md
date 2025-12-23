# Layer-Domain Control: A Mechanistic Study of Domain Specialization

##  Objective

This project investigates a fundamental question in the study of Large Language Models: **How does a single, monolithic network pivot its internal mechanisms to handle diverse and specialized domains like programming, medicine, and finance?**

Our goal is to move beyond correlational analysis and build a causal, layer-level map of a model's functional architecture. We achieve this by introducing the **Layer-Domain Control (LDC)** framework, a unified methodology that synthesizes evidence from representational analysis, adaptational analysis, and causal intervention.

This document focuses on the **Adaptational Analysis**, which aims to identify and quantify which parameters change the most when the model is fine-tuned on a new domain, thereby revealing the primary locus of learning.

---

##  Fine-Tuning Methodology

To map where domain-specific knowledge is written during adaptation, we employed a systematic, LoRA-based fine-tuning methodology across four models (**Llama 3.2 3B, Llama 3.2 1B, Gemma 3 4B, and Gemma 3 1B**) and six distinct domains (**C++, Python, Math, Science, Finance, and Medical**).

### Parameter-Efficient Adaptation with LoRA

We use Low-Rank Adaptation (LoRA) as a principled and efficient proxy for full fine-tuning. Instead of updating a large pre-trained weight matrix $W$, LoRA introduces a low-rank decomposition of the weight update $\Delta W$. For a given layer $\ell$, the update is defined as:

$$
\Delta W_\ell = \frac{\alpha}{r} B_\ell A_\ell
$$

where $A_\ell \in \mathbb{R}^{r \times m}$ and $B_\ell \in \mathbb{R}^{n \times r}$ are the low-rank adapter matrices, $r$ is the rank, and $\alpha$ is a scaling factor. Only the parameters of $A_\ell$ and $B_\ell$ are trained. Research indicates that the matrix representing the change in weights during full fine-tuning tends to have a low intrinsic rank, making LoRA a valid and principled approximation of the full fine-tuning process.


## Why Lora is a Valid Proxy for full training 

Instead of modifying a huge, pre-trained weight matrix `W` (which has millions of parameters), LoRA keeps `W` frozen. It learns the *change* to the weights, `ΔW`, by representing this change as the product of two much smaller, "low-rank" matrices, `A` and `B`. During a forward pass, the model's output is calculated as `h = Wx + BAx`. Only `A` and `B` are trained. Because `A` and `B` are tiny compared to `W`, we are training only a fraction of a percent of the total parameters, leading to massive savings in memory and computation time.

**Why This Shows Similar Results to Full Training:** This experiment's methodology is grounded in a key research finding known as the **"low-rank hypothesis."** Studies have shown that when large pre-trained models are fully fine-tuned, the matrix representing the change in weights (`ΔW = W_final - W_initial`) tends to have a very low "intrinsic rank." This means the complex update across millions of parameters can be effectively approximated without losing significant information. LoRA is explicitly designed to create such a low-rank update. Therefore, by constraining the update to be low-rank, LoRA is not just an efficiency hack; it's a **principled approximation of the full fine-tuning process.** Analyzing where LoRA applies its largest changes gives us a strong and valid insight into which layers would have been most modified during a full fine-tuning run.


### Quantifying Adaptation: The Frobenius Norm

Our primary metric for quantifying the magnitude of change in a layer is the **Frobenius norm** of the effective weight update, $S_\ell$. A higher value for $S_\ell$ indicates that the parameters in that layer are a primary site for storing new, domain-specific computation learned during adaptation.

$$
S_\ell = \|\Delta W_\ell\|_F
$$

For a comprehensive view, we aggregate the norms of all adapted components within a single Transformer block (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj` for attention) by summation: $S_{\ell}^{\text{block}} = \sum_{t \in T_\ell} \|\Delta W_t\|_F$.

### A Two-Stage Experimental Design

Our investigation followed a two-stage process to first map and then validate the locus of adaptation.

#### Stage 1: Comprehensive Adaptational Mapping

In the first stage, we conducted a broad, component-wise analysis for each domain. We applied LoRA adapters under three distinct regimes to understand the division of labor:

* **Attention-Only**: Adapters were applied exclusively to the attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) in every layer.
* **MLP-Only**: Adapters were applied exclusively to the MLP projection matrices (`gate_proj`, `up_proj`, `down_proj`) in every layer.
* **Full Model (All)**: Adapters were applied to all attention and MLP components simultaneously, serving as a baseline for full-model adaptation.

#### Stage 2: Targeted Validation

The adaptational map generated in Stage 1 was then validated. We analyzed the $S_\ell$ values from the full-model runs to identify the layers that underwent the most significant changes. Based on this ranking, we designed a new set of targeted fine-tuning configurations, including "Soloist Tuning" (adapting only the single top-ranked layer) and "Ensemble Tuning" (adapting the top-3 layers).

---

## 📊 Results and Discussion

### Adaptational Analysis Points to MLP Layers

Our initial adaptational mapping revealed a clear and consistent pattern across all models and domains: **the optimizer dedicates the vast majority of its updates to the MLP layers.** As shown in the figures below, the magnitude of weight change in MLP-only fine-tuning is substantially higher than in attention-only fine-tuning and closely tracks the MLP changes observed in full-model tuning. This provides strong evidence that MLP layers are the primary locus where new, domain-specific computation is written and stored.

#### Llama 3.2 3B - Layer-wise Weight Changes

| C++                                                                                                                      | Python                                                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| ![Llama 3B C++](inside_llms/phase1/fine_tune_new/full_new/llama_3b/fig_norm_llama_3b_cpp.png)                                  | ![Llama 3B Python](inside_llms/phase1/fine_tune_new/full_new/llama_3b/fig_norm_llama_3b_python.png)                                |
| **Math** | **Medical** |
| ![Llama 3B Math](inside_llms/phase1/fine_tune_new/full_new/llama_3b/fig_norm_llama_3b_math.png)                                | ![Llama 3B Medical](inside_llms/phase1/fine_tune_new/full_new/llama_3b/fig_norm_llama_3b_medical.png)                              |
| **Science** | **Finance** |
| ![Llama 3B Science](inside_llms/phase1/fine_tune_new/full_new/llama_3b/fig_norm_llama_3b_science.png)                          | ![Llama 3B Finance](inside_llms/phase1/fine_tune_new/full_new/llama_3b/fig_norm_llama_3b_finance.png)                              |

#### Llama 3.2 1B - Layer-wise Weight Changes

| C++                                                                                                                      | Python                                                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| ![Llama 1B C++](inside_llms/phase1/fine_tune_new/full_new/llama_1b/fig_norm_llama_1b_cpp.png)                                  | ![Llama 1B Python](inside_llms/phase1/fine_tune_new/full_new/llama_1b/fig_norm_llama_1b_python.png)                                |
| **Math** | **Medical** |
| ![Llama 1B Math](inside_llms/phase1/fine_tune_new/full_new/llama_1b/fig_norm_llama_1b_math.png)                                | ![Llama 1B Medical](inside_llms/phase1/fine_tune_new/full_new/llama_1b/fig_norm_llama_1b_medical.png)                              |
| **Science** | **Finance** |
| ![Llama 1B Science](inside_llms/phase1/fine_tune_new/full_new/llama_1b/fig_norm_llama_1b_science.png)                          | ![Llama 1B Finance](inside_llms/phase1/fine_tune_new/full_new/llama_1b/fig_norm_llama_1b_finance.png)                              |

#### Gemma 3 4B - Layer-wise Weight Changes

| C++                                                                                                                      | Python                                                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| ![Gemma 4B C++](inside_llms/phase1/fine_tune_new/full_new/gemma_4b/fig_norm_gemma_4b_cpp.png)                                  | ![Gemma 4B Python](inside_llms/phase1/fine_tune_new/full_new/gemma_4b/fig_norm_gemma_4b_python.png)                                |
| **Math** | **Medical** |
| ![Gemma 4B Math](inside_llms/phase1/fine_tune_new/full_new/gemma_4b/fig_norm_gemma_4b_math.png)                                | ![Gemma 4B Medical](inside_llms/phase1/fine_tune_new/full_new/gemma_4b/fig_norm_gemma_4b_medical.png)                              |
| **Science** | **Finance** |
| ![Gemma 4B Science](inside_llms/phase1/fine_tune_new/full_new/gemma_4b/fig_norm_gemma_4b_science.png)                          | ![Gemma 4B Finance](inside_llms/phase1/fine_tune_new/full_new/gemma_4b/fig_norm_gemma_4b_finance.png)                              |

#### Gemma 3 1B - Layer-wise Weight Changes

| C++                                                                                                                      | Python                                                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| ![Gemma 1B C++](inside_llms/phase1/fine_tune_new/full_new/gemma_1b/fig_norm_gemma_1b_cpp.png)                                  | ![Gemma 1B Python](inside_llms/phase1/fine_tune_new/full_new/gemma_1b/fig_norm_gemma_1b_python.png)                                |
| **Math** | **Medical** |
| ![Gemma 1B Math](inside_llms/phase1/fine_tune_new/full_new/gemma_1b/fig_norm_gemma_1b_math.png)                                | ![Gemma 1B Medical](inside_llms/phase1/fine_tune_new/full_new/gemma_1b/fig_norm_gemma_1b_medical.png)                              |
| **Science** | **Finance** |
| ![Gemma 1B Science](inside_llms/phase1/fine_tune_new/full_new/gemma_1b/fig_norm_gemma_1b_science.png)                          | ![Gemma 1B Finance](inside_llms/phase1/fine_tune_new/full_new/gemma_1b/fig_norm_gemma_1b_finance.png)                              |

### Validating the Map with Targeted Fine-Tuning

If the layers with the largest parameter deltas are indeed the most important, then fine-tuning only these layers should yield a parameter-efficient performance gain. We tested this by fine-tuning only the top-1 and top-3 layers identified by our analysis and evaluating their performance on domain-specific perplexity tasks (normalized between 0 and 1).

The results confirm our hypothesis. As shown below for Llama 3.2 3B, targeted fine-tuning of just a few layers often achieves performance comparable to—and sometimes exceeding—that of fine-tuning the entire model, despite using a fraction of the parameters. This demonstrates that our adaptational map is not merely descriptive but predictive.

| Domain  | PT   | MLP  | Attn | Both | Top-1 MLP | Top-1 Attn | Top-3 MLP | Top-3 Attn |
| :------ | :--- | :--- | :--- | :--- | :-------- | :--------- | :-------- | :--------- |
| Math    | 0.07 | 0.03 | 0.00 | 0.02 | 0.08      | 0.07       | **0.12** | 0.03       |
| Science | 0.82 | 0.73 | 0.80 | 0.62 | 0.76      | 0.76       | **0.86** | 0.66       |
| CPP     | 0.31 | 0.06 | 0.05 | 0.04 | 0.01      | 0.19       | 0.02      | **0.41** |
| Python  | 0.60 | 0.16 | 0.36 | 0.02 | 0.14      | **0.57** | 0.19      | 0.56       |
| Finance | 0.16 | 0.06 | 0.05 | 0.02 | 0.05      | 0.05       | **0.08** | 0.06       |
| Medical | 0.58 | 0.91 | 0.84 | 0.89 | **0.93** | 0.30       | 0.91      | 0.54       |

### Granular Adaptational Norm Analysis

To further dissect the dynamics of adaptation, we analyzed the Frobenius norms of the top-3 layers under different training regimes: the baseline **Full Run**, a targeted **Ensemble** run, and an isolated **Solo** run. The tables below summarize these norms, averaged across the top-3 components, for each model. This granular view reinforces that while MLP norms are consistently larger, targeted adaptation often elicits an even stronger response from these key layers compared to the baseline full run.

#### Llama 3.2 3B - Aggregated Norms

| Domain  | Component Group               | Avg. Full Run Norm | Avg. Ensemble Norm | Top Solo Run Norm |
| :------ | :---------------------------- | :----------------- | :----------------- | :---------------- |
| CPP     | Top-3 MLP Components (Avg.)   | 1.019e+02          | 1.287e+02          | 1.651e+02         |
|         | Top-3 Attn Components (Avg.)  | 7.042e+01          | 9.357e+01          | 1.149e+02         |
| Finance | Top-3 MLP Components (Avg.)   | 9.463e+01          | 6.990e+01          | 9.945e+01         |
|         | Top-3 Attn Components (Avg.)  | 5.464e+01          | 4.687e+01          | 6.056e+01         |
| Math    | Top-3 MLP Components (Avg.)   | 1.007e+02          | 1.360e+02          | 1.676e+02         |
|         | Top-3 Attn Components (Avg.)  | 6.580e+01          | 8.377e+01          | 9.766e+01         |
| Medical | Top-3 MLP Components (Avg.)   | 9.545e+01          | 1.239e+02          | 1.560e+02         |
|         | Top-3 Attn Components (Avg.)  | 9.134e+01          | 9.702e+01          | 1.181e+02         |
| Python  | Top-3 MLP Components (Avg.)   | 1.010e+02          | 1.311e+02          | 1.744e+02         |
|         | Top-3 Attn Components (Avg.)  | 6.978e+01          | 9.599e+01          | 1.250e+02         |
| Science | Top-3 MLP Components (Avg.)   | 1.019e+02          | 1.343e+02          | 1.660e+02         |
|         | Top-3 Attn Components (Avg.)  | 8.013e+01          | 9.999e+01          | 1.145e+02         |

#### Llama 3.2 1B - Aggregated Norms

| Domain  | Component Group               | Avg. Full Run Norm | Avg. Ensemble Norm | Top Solo Run Norm |
| :------ | :---------------------------- | :----------------- | :----------------- | :---------------- |
| CPP     | Top-3 MLP Components (Avg.)   | 1.131e+02          | 1.508e+02          | 1.944e+02         |
|         | Top-3 Attn Components (Avg.)  | 8.660e+01          | 1.150e+02          | 1.389e+02         |
| Finance | Top-3 MLP Components (Avg.)   | 1.063e+02          | 8.506e+01          | 1.201e+02         |
|         | Top-3 Attn Components (Avg.)  | 6.711e+01          | 5.694e+01          | 7.221e+01         |
| Math    | Top-3 MLP Components (Avg.)   | 1.151e+02          | 1.607e+02          | 2.025e+02         |
|         | Top-3 Attn Components (Avg.)  | 8.405e+01          | 1.062e+02          | 1.209e+02         |
| Medical | Top-3 MLP Components (Avg.)   | 1.116e+02          | 1.457e+02          | 1.798e+02         |
|         | Top-3 Attn Components (Avg.)  | 1.139e+02          | 1.196e+02          | 1.402e+02         |
| Python  | Top-3 MLP Components (Avg.)   | 1.123e+02          | 1.535e+02          | 2.042e+02         |
|         | Top-3 Attn Components (Avg.)  | 8.578e+01          | 1.173e+02          | 1.493e+02         |
| Science | Top-3 MLP Components (Avg.)   | 1.189e+02          | 1.610e+02          | 2.009e+02         |
|         | Top-3 Attn Components (Avg.)  | 1.024e+02          | 1.278e+02          | 1.446e+02         |

#### Gemma 3 4B - Aggregated Norms

| Domain  | Component Group               | Avg. Full Run Norm | Avg. Ensemble Norm | Top Solo Run Norm |
| :------ | :---------------------------- | :----------------- | :----------------- | :---------------- |
| CPP     | Top-3 MLP Components (Avg.)   | 7.481e+01          | 8.510e+01          | 9.509e+01         |
|         | Top-3 Attn Components (Avg.)  | 4.523e+01          | 5.179e+01          | 6.092e+01         |
| Finance | Top-3 MLP Components (Avg.)   | 3.211e+01          | 3.883e+01          | 4.720e+01         |
|         | Top-3 Attn Components (Avg.)  | 2.398e+01          | 2.806e+01          | 3.566e+01         |
| Math    | Top-3 MLP Components (Avg.)   | 6.152e+01          | 7.033e+01          | 7.748e+01         |
|         | Top-3 Attn Components (Avg.)  | 3.345e+01          | 3.862e+01          | 4.418e+01         |
| Medical | Top-3 MLP Components (Avg.)   | 7.913e+01          | 8.882e+01          | 1.060e+02         |
|         | Top-3 Attn Components (Avg.)  | 4.881e+01          | 5.361e+01          | 6.759e+01         |
| Python  | Top-3 MLP Components (Avg.)   | 7.612e+01          | 8.496e+01          | 9.706e+01         |
|         | Top-3 Attn Components (Avg.)  | 4.755e+01          | 5.305e+01          | 6.187e+01         |
| Science | Top-3 MLP Components (Avg.)   | 8.339e+01          | 9.547e+01          | 1.049e+02         |
|         | Top-3 Attn Components (Avg.)  | 4.698e+01          | 5.223e+01          | 5.652e+01         |

#### Gemma 3 1B - Aggregated Norms

| Domain  | Component Group               | Avg. Full Run Norm | Avg. Ensemble Norm | Top Solo Run Norm |
| :------ | :---------------------------- | :----------------- | :----------------- | :---------------- |
| CPP     | Top-3 MLP Components (Avg.)   | 4.315e+01          | 5.039e+01          | 6.484e+01         |
|         | Top-3 Attn Components (Avg.)  | 2.451e+01          | 2.822e+01          | 3.337e+01         |
| Finance | Top-3 MLP Components (Avg.)   | 2.478e+01          | 2.891e+01          | 3.953e+01         |
|         | Top-3 Attn Components (Avg.)  | 1.691e+01          | 1.956e+01          | 3.240e+01         |
| Math    | Top-3 MLP Components (Avg.)   | 4.022e+01          | 4.570e+01          | 5.823e+01         |
|         | Top-3 Attn Components (Avg.)  | 2.003e+01          | 2.292e+01          | 2.922e+01         |
| Medical | Top-3 MLP Components (Avg.)   | 4.811e+01          | 5.544e+01          | 7.106e+01         |
|         | Top-3 Attn Components (Avg.)  | 2.955e+01          | 3.401e+01          | 4.053e+01         |
| Python  | Top-3 MLP Components (Avg.)   | 4.297e+01          | 4.926e+01          | 6.502e+01         |
|         | Top-3 Attn Components (Avg.)  | 2.501e+01          | 2.846e+01          | 3.237e+01         |
| Science | Top-3 MLP Components (Avg.)   | 4.973e+01          | 5.627e+01          | 6.923e+01         |
|         | Top-3 Attn Components (Avg.)  | 2.516e+01          | 2.830e+01          | 3.364e+01         |
