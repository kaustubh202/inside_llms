---
icon: hand-wave
---

# inside_llms

Large Language Models are the talk of the town, and as they gain more ground, it has become increasingly difficult and important to understand what's going on inside them.
Understanding a large language model often feels like staring at a city skyline at night, there are plenty of lights, but not much sense of who’s inside which room.
We’re a small group of undergraduate researchers who wanted to map specific “rooms” inside one large language model, trying to find layers that lean toward Mathematical Detail, others that prefer Programming Languages, and so on, without enterprise-grade GPUs. Trying to figure out where exactly domain-specific knowledge is stored in an LLM.

Our approach so far

-   **Model in scope**: _Llama 3.2-3B_ (27 transformer layers), chosen because it fits on free Kaggle GPUs. We'll delve into more models over time. (We have limited resources as "unsupervised" undergrads)
-   **Domain sets**: short corpora for C++, Python, mathematics (thinking + solutions), and physics explanations.
-   **Experiments**: forward-pass profiling, lightweight fine-tunes, logistic probes, zero-out tests, and a few interpretability touches. Details of each experiment can be found on its dedicated page.

What you’ll find on this site

1. **Formal Objective** – the page spelling out the research question and evaluation criteria.
2. **Mech Interp Experiments** – the step-by-step experiments we performed with plots and takeaway notes.
3. **Concept Vectors** – the concept of trying to represent domain knowledge as directions in hidden space.
4. **Conclusion** – what the results suggest about modularity and future compression work.
5. **Team** – faces, emails, and other details about us.

Our gitbook link: https://inside-llms.gitbook.io/surgical-domain-discovery

We’ll keep updating as new tests run (or crash). Feedback and replication attempts are welcome; open a pull request or mail us anytime.

_Inside LLMs_ is a work in progress, shared openly so the conversation can start before the finish line.
