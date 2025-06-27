# Formal Objective

When a human acquires a new subject, say literature one day, mathematics the next, and general news some other day, the related ideas settle into distinct areas of the brain.  
We ask whether something similar happens inside a large language model (LLM).

**Hypothesis**  
In an LLM, there exist identifiable components - layers, heads, or computational sub-graphs - that carry most of the workload for a given domain (e.g. Python programming).  
If those components can be isolated, we can:

* keep the **domain-critical part** intact,  
* replace the **domain-neutral remainder** with a lighter surrogate,  
* obtain a smaller model that maintains high performance on its chosen domain **without full re-training**.

---

## Scope of the study

| Item | Choice |
|------|--------|
| Base model | Llama 3.2-3B (27 transformer layers) |
| Domains analysed | C++, Python, Mathematics (Thinking & Solutions), Physics explanations |
| Resources | Free Kaggle GPUs; all experiments sized to fit this limit |

---

## Research questions

1. **Localisation**  
   Which layers (or heads) carry the strongest domain signal?
2. **Redundancy**  
   Is domain knowledge concentrated or spread across many layers?
3. **Compression potential**  
   After replacing the low-relevance layers with lighter surrogates, how much size and inference time can we save while staying within a small accuracy drop?

---

## Formal statement

Let $M$ be the original 27-layer model.  
Let $d$ be a domain from our set.  
Let $S_d$ be the subset of layers retained for domain $d$.
Let $\text{Perf}(M,d)$ be a domain metric (perplexity or task score).  

The goal is to find the smallest $S_d$ such that:

$$
\text{Perf}(M_{S_d}, d) \ge (1 - \varepsilon)\, \text{Perf}(M, d),
$$

with $\varepsilon$ set to 0.05 (i.e., ≤ 5% degradation).  
Layer relevance is estimated through **probe separability**, **zero-out tests**, and **finetune-delta measurements**; details appear in the experiment pages.

---

This page sets up what we want to prove; the sections in the sidebar document how close we get.
