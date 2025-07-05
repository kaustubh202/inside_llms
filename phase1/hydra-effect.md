# Hydra Effect
In this experiment, we utilize the concept of Hydra Effect to analyze the domain knowledge stored in layers from a different prespective. 
## Theory
The Hydra Effect was coined first in a 2023 paper by Google DeepMind team where they investigate the internal structure of LLM computations using causal analysis. They show how LLMs adapt to ablations of one layer by compensating for the change by counterbalancing the activations of subsequent layers. This effect is called the Hydra effect since it resembles the behaviour of the Hydra monster, in which other heads grow in importance when one set of heads is cut off. They show that LLMs exhibit not only just redundancy but also active self-repairing computations. To create a representation of how a layer adjusts its outputs for self-repair, the researchers use unembedding logits.

First we need to define some terminology. Suppose a Transformer model has $L$ layers. It aims to predict the next token $x_{t+1}$ given the sequence of input tokens $x_{\leq t} = (x_1, x_1, ...., x_t)$ using a function $f_{\theta}$ which is our transformer. 
$$
\begin{equation*}
    \begin{split}
    p(x_{t+1} | x_{\leq t}) & = f_{\theta}(x_{\leq t}) = \text{SoftMax}(\pi_t(x_{\leq t}))\\
    \end{split}
\end{equation*}
$$
Here the pre-softmax terms $\pi$ are called logits. Our whole analysis shall be based on these values for the last position in the sequence. 
$$
\begin{equation*}
    \begin{split}
        \pi_t &= \text{RMSNorm}(z_t^L)W_U \\
        z_t^l &= z_t^{l-1} + a_t^l + m_t^l \\
        a_t^l &= \text{Attn}(z_{\leq t}^{l-1}) \\
        m_t^l &= \text{MLP}(z_t^{l-1}) \\
    \end{split}
\end{equation*}
$$
For a standard Transformer, the process of unembedding, i.e., obtaining the logits is performed only on the output of the last layer to get the final prediction. But what if we did this process on earlier layers as well?

This process of applying the unembedding mechanism on an earlier layer is called logit lens. This shows the model's current understanding of the inputs till that layer. Suppose $\tilde \pi_t^l$ is the logit distribution for a particular layer $l$. We can apply the learned Unembedding Matrix $W_U$ and RMSNorm layer from the last layer on the output of this layer to compute this distribution.
$$
\begin{equation*}
    \begin{split}
        \tilde \pi_t^l &= \text{RMSNorm}(z_t^l)W_U \\
    \end{split}
\end{equation*}
$$
It should be noted that for any kind of analysis on logits, it is necessary to centre them since their absolute values are not important. We only want to see the relative shifts that the layer assigns to different tokens. 
$$
\begin{equation*}
    \begin{split}
        \^{\pi}_t = \tilde \pi_t - \mu_{\pi};\quad \mu_{\pi} = \frac{1}{V} \sum_i^V [\tilde \pi_t]_i \\
    \end{split}
\end{equation*}
$$
The metric to check the layer's impact on the final output is the value of this centred logit for the maximum-likelihood token at the final token position for the last layer. 
$$
\begin{equation*}
    \begin{split}
        \Delta_{\text{unembed}, l} = \^{\pi}_t(a^l)_i ; \quad i = \argmax_j[\pi_t^L]_j
    \end{split}
\end{equation*}
$$
Next, we look at the process of ablations, which means surgically removing the acitvations of one layer for a particular input and replacing it with a different value. For example, zero-out ablation means removing the layer's output and setting all values to zero and putting it back. 
For mathematically denoting the ablation we need some additional notation that is standard in the field of causality. The researchers indicate replacing activations of a layer $a_t^l$ with another using the do(.) operation. Then the intervention on the output of a layer A_t^l with output from another input $x'$ can be denoted as, 
$$
\begin{equation*}
    \begin{split}
        \pi_t(x_{\leq t}| \text{do}(A^l_t = \tilde{a}_t^l)) = \pi_t(x_{\leq t}| \text{do}(A^l_t = a_t^l(x'_{\leq t})))
    \end{split}
\end{equation*}
$$
By ablating a layer, we can see how the subsequent $\Delta_\text{unembed}$ changes for other layers. 