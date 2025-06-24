---
icon: hand-wave
---

# inside\_llms

Large Language Models are the talk of the town, and as they're gaining more grounds, it's become increasingly difficult and important to understand what's going inside them.
Understanding a large language model often feels like staring at a city skyline at night, there are plenty of lights, but not much sense of who’s inside which room.  
We’re a small group of undergraduate researchers who wanted to map specific “rooms” inside one model—layers that lean toward Python, others that prefer physics, and so on—without enterprise-grade GPUs. Basically, trying to figure out where exactly domain-specific knowledge is stored in an LLM.

Our approach so far

* **Model in scope**: *Llama 3.2-3B* (27 transformer layers), chosen because it fits on free Kaggle GPUs. (We have limited resources as "unsupervised" undergrads)
* **Domain sets**: short corpora for C++, Python, mathematics (questions + solutions), and physics explanations.  
* **Experiments**: forward-pass profiling, lightweight fine-tunes, logistic probes, zero-out tests, and a few interpretability touches. Each experiment lives in its own page in the sidebar.  
* **Early pattern**- mid-stack layers (around 10–15) show the most separability; some heads are versatile across programming languages, others are surprisingly single-minded.

What you’ll find on this site

1. **Formal Objective** – one page spelling out the research question and evaluation criteria.  
2. **Phase 1 notebooks** – the step-by-step experiments with plots and takeaway notes.  
3. **Concept Vectors** – how we’re trying to represent domain knowledge as directions in hidden space.  
4. **Conclusion** – what the results suggest about modularity and future compression work.  
5. **Team** – faces, emails, and the tools we keep open in too many browser tabs.

We’ll keep updating as new tests run (or crash). Feedback and replication attempts are welcome—open an issue or pull request any time.

*Inside LLMs* is a work in progress, shared openly so the conversation can start before the finish line.
