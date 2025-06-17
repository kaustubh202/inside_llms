# Probe Separability Experiment Using V-Usable Information

This Readme contains three experiments to evaluate the **separability** of different domains** of the **LLaMA-3.2-3B model** using **v-usable information** . Each experiment probes a specific component of the model’s transformer layers: **Multi-Layer Perceptrons (MLPs)**, **self-attention outputs**, or **residual stream outputs**, to assess their ability to distinguish between three datasets: **CNN Daily Mail**, **IMDb**, and **Yelp Reviews**. Achieveing a high v-usable information in some of the layers can be supportive of surgical domain extraction.
## Objective of the Experiment

The objective of these experiments is to measure how well features extracted from different components of the LLaMA-3.2-3B model’s transformer layers—**MLP**, **attention**, and **residual stream**—can separate samples from three distinct datasets (CNN Daily Mail, IMDb, and Yelp Reviews) using **v-usable information**. **V-usable information**, expressed in bits, quantifies the amount of information a feature provides for distinguishing between classes (here, the three datasets)) in a **logistic regression probe**. By probing each component separately (i.e. MLP layers,Attention layers,Residual Stream activations), we aim to understand their individual contributions to dataset-specific representations and identify which layers encode the most domain-specific information. This insight can help in surgical domain extraction task by informing us of the separability between domains present.
Note that citing from https://arxiv.org/abs/2402.16061 it is One thing is we can’t directly judge based on accuracy hence we also used V-usable information. V-usable information reflects the ease with which a model family V can predict the output Y given specific input R effectively 

**Probe separability** refers to the ability of a linear probe (logistic regression) trained on extracted features to classify samples by their dataset of origin. A higher v-usable-usable bit score indicates better separability, meaning the features contain more information relevant to distinguishing the datasets. The three experiments isolate:
- **MLP Layers**: Non-linear transformations of attention outputs.
- **Attention Layers**: Contextual relationships captured via self-attention mechanisms.
- **Residual Stream**: Cumulative representations combining attention and MLP outputs with prior layer information.

---

## Methodology

### Model and Environment
- **Model**: LLaMA-3.2-3B, a 3-billion-parameter transformer model from Meta AI, loaded in `float16` precision to reduce memory usage.
- **Framework**: PyTorch with Hugging Face Transformers for model and tokenizer, scikit-learn for logistic regression, and Matplotlib for visualization.
- 
### V-Usable Information
**V-usable information** (in bits) measures the information content of features for a classification task.
V-usable information quantifies how much an input helps a model family  predict an output measured in bits. It shows the reduction in uncertainty about  Y  when using the input.
In Simple Terms: V-usable information is how much easier it is to predict ( Y ) (e.g., dataset labels) with the input (e.g., LLaMA-3.2-3B layer features). Higher bits mean better separability. In your experiments, a logistic regression probe estimates this by testing how well MLP, attention, or residual features distinguish CNN, IMDb, and Yelp datasets.


- Higher v-usable bits indicate better separability (more information for distinguishing datasets).

### Datasets
- **Datasets**: Three processed JSON files containing text samples:
  - **CNN Daily Mail**: News articles.
  - **IMDb**: Movie reviews.
  - **Yelp Reviews**: Business reviews.
- **Sample Size**: 1,000 samples per dataset (total 3,000 samples), enforced by `max_per_domain=1000` in `DomainIterableDataset`, to balance speed and statistical reliability.
- **Preprocessing**: Texts are tokenized with LLaMA’s tokenizer, padded/truncated to 512 tokens.

### Key Parameters
- **Layers Probed**: 22 layers (indices `[0, 1, 2, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]`) out of LLaMA’s 26 layers.
- **Batch Size**: 8 samples per batch for feature extraction.
- **Sequence Pooling**: Mean pooling over sequence dimension to reduce activations to `[batch_size, hidden_size]`.
- **Probe**: Multinomial logistic regression with L2 penalty (`C=1.0`), `saga` solver, 20% validation split.

### Workflow
1. Load LLaMA-3.2-3B and tokenizer, stream datasets.
2. Extract features (MLP, attention, or residual) per layer using forward hooks.
3. Train logistic regression probe on features to classify dataset origin.
4. Compute accuracy and v-usable bits.
5. Plot v-usable bits vs. layer index.

---

## Implementation

The three experiments share a common codebase, with differences only in the type of activations probed (MLP, attention, or residual stream).
### Common Components

#### Imports and Setup
```python
import json, gc, math, os, random, itertools, pathlib, functools
from typing import List, Dict, Iterable, Tuple
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
token = "hf_xxxxxxxx token "

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
model = (AutoModelForCausalLM
         .from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                          low_cpu_mem_usage=True, token=token)
         .to(DEVICE)
         .eval())
tokenizer.pad_token = tokenizer.eos_token
```
**Explanation**: Imports libraries for data handling (`json`, `pathlib`), computation (`torch`, `numpy`), visualization (`matplotlib`), and modeling (`transformers`, `sklearn`). Sets up the LLaMA-3.2-3B model in `float16` on GPU, with its tokenizer configured to use the end-of-sequence token as padding. The `tqdm` library provides progress bars for feature extraction.

#### Dataset Streaming
```python
class DomainIterableDataset(IterableDataset):
    def __init__(self, json_paths: List[str], max_per_domain: int = 10_000, shuffle: bool = True):
        self.paths = [pathlib.Path(p) for p in json_paths]
        self.max_per_domain = max_per_domain
        self.shuffle = shuffle
        self.domain2id = {p: i for i, p in enumerate(self.paths)}

    def _sample_texts(self, file_path: pathlib.Path) -> Iterable[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            texts = json.load(f)
        if self.shuffle:
            random.shuffle(texts)
        return texts[:self.max_per_domain]

    def __iter__(self):
        files = self.paths
        if self.shuffle:
            random.shuffle(files)
        for fp in files:
            d_id = self.domain2id[fp]
            for txt in self._sample_texts(fp):
                yield txt, d_id

DOMAIN_JSONS = [
    "/kaggle/input/my-datasets-for-llama-profiling/cnn_dailymail_processed.json",
    "/kaggle/input/my-datasets-for-llama-profiling/imdb_processed.json",
    "/kaggle/input/my-datasets-for-llama-profiling/yelp_review_full_processed.json",
]
```
**Explanation**: Defines `DomainIterableDataset` to stream text samples from three JSON files (CNN, IMDb, Yelp), yielding tuples of (text, domain_id). Each dataset is limited to `max_per_domain=1000` samples, shuffled for randomness. The `DOMAIN_JSONS` list specifies Kaggle dataset paths.

#### Activation Grabber (Common Parts)
```python
class ActivationGrabber:
    def __init__(self, layer_idx: int, seq_pool: str = "mean"):
        self.L = layer_idx
        self.seq_pool = seq_pool
        self.clear()

    def _save(self, name):
        def hook(_, __, out):
            if isinstance(out, tuple):
                out = out[0]
            self.buffers[name] = out.detach()
        return hook

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_pool == "mean":
            return x.mean(dim=1)
        if self.seq_pool == "first":
            return x[:, 0]
        raise ValueError(self.seq_pool)

    def clear(self):
        self.buffers: Dict[str, torch.Tensor] = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
```
**Explanation**: Implements `ActivationGrabber`, a context manager for registering forward hooks on model layers to capture activations. Initializes with a layer index and pooling method (`mean` by default). The `_save` method creates hooks to store detached tensors, `_pool` reduces sequence dimensions (e.g., `[batch_size, seq_len, hidden_size]` to `[batch_size, hidden_size]`), and `clear`/`__exit__` manage memory by clearing buffers and removing hooks.

#### Feature Collection (Common Parts)
```python
def collect_layer_features(layer: int, dataloader: DataLoader, n_examples: int = None):
    feats, labels = [], []

    with torch.no_grad(), ActivationGrabber(layer) as grabber:
        for batch_txts, batch_domains in tqdm(dataloader, desc=f"Layer {layer}"):
            toks = tokenizer(batch_txts,
                             return_tensors="pt", padding=True, truncation=True,
                             max_length=512).to(DEVICE)
            _ = model(**toks)
            feats.append(feat.astype(np.float32))
            labels.append(np.asarray(batch_domains))

    X = np.concatenate(feats, 0)
    y = np.concatenate(labels, 0)
    return X, y
```
**Explanation**: Collects features for a given layer by iterating over a `DataLoader`. Tokenizes texts (max 512 tokens), runs a forward pass with `torch.no_grad()` to save memory, and appends pooled activations and domain labels. Concatenates features into a matrix `X` (`[n_samples, hidden_size]`) and labels into `y` (`[n_samples]`). The `feat` placeholder is replaced by specific activation types in each experiment.

#### Logistic Regression Probe
```python
def train_logreg_probe(X, y, val_split=0.2, C=1.0) -> Dict[str, float]:
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    n_val = int(len(y) * val_split)
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr, y_tr = X[n_val:], y[n_val:]

    probe = LogisticRegression(
        penalty="l2", C=C, max_iter=1000,
        multi_class="multinomial", solver="saga", n_jobs=-1, verbose=0)
    probe.fit(X_tr, y_tr)

    y_pred = probe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    logp = probe.predict_log_proba(X_val)
    n_cls = len(np.unique(y))
    H_emp = -log_loss(y_val, probe.predict_proba(X_val),
                      labels=np.arange(n_cls), normalize=True)
    v_bits = (math.log(n_cls) + H_emp) / math.log(2)

    return dict(acc=acc, v_usable_bits=v_bits, probe=probe)
```
**Explanation**: Trains a multinomial logistic regression probe on features `X` and labels `y`. Shuffles data, splits 20% for validation, and fits the probe with L2 regularization. Computes validation accuracy and v-usable bits using cross-entropy loss, returning a dictionary with metrics and the trained probe.

#### Data Collation and Probing Pipeline
```python
def collate(batch):
    texts, labels = zip(*batch)
    return list(texts), list(labels)

def probe_layer(layer_idx: int, json_paths: List[str], max_per_domain=10_000, batch_size=8, n_examples=None):
    ds = DomainIterableDataset(DOMAIN_JSONS, max_per_domain=1000, shuffle=True)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate, shuffle=False, num_workers=0)

    X, y = collect_layer_features(layer_idx, dl, n_examples=n_examples)
    print("unique labels:", np.unique(y, return_counts=True))

    stats = train_logreg_probe(X, y)

    result = {
        "layer": layer_idx,
        "n_examples": len(y),
        "acc": stats["acc"],
        "v_bits": stats["v_usable_bits"],
    }
    print(result)
    del X, y
    gc.collect(); torch.cuda.empty_cache()
    return result
```
**Explanation**: The `collate` function converts batches of (text, label) tuples into lists for tokenization. The `probe_layer` function orchestrates the pipeline: creates a dataset with `max_per_domain=1000`, loads it in batches of 8, collects features, trains the probe, and returns metrics. Prints label counts and results, and clears memory to prevent GPU memory leaks.

#### Main Loop and Plotting
```python
layers_to_probe = [0, 1, 2, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
results = []

for L in layers_to_probe:
    res = probe_layer(L, DOMAIN_JSONS,
                      max_per_domain=5000,
                      batch_size=8,
                      n_examples=None)
    results.append(res)

def plot_v_bits(results):
    layers = [r["layer"] for r in results]
    v_bits = [r["v_bits"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(layers, v_bits, marker='o')
    plt.xlabel('Layer Index')
    plt.ylabel('V-Usable Bits')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_v_bits(results)
print("Done!")
```
**Explanation**: Iterates over specified layers, probing each and storing results. The `plot_v_bits` function visualizes v-usable bits vs. layer indices using Matplotlib. The plot configuration (label, title) varies per experiment. Prints “Done!” upon completion.

### Differing Components

The three experiments differ in three areas: (1) the activation hook in `ActivationGrabber.__enter__`, (2) the activation retrieval in `ActivationGrabber.pooled` and `collect_layer_features`, and (3) the plot configuration in `plot_v_bits`. Below, we present the differing code blocks for each experiment with explanations.

#### Activation Hook (`ActivationGrabber.__enter__`)
- **MLP Experiment**:
  ```python
  def __enter__(self):
      layer = model.model.layers[self.L]
      self.handles = [
          layer.mlp.register_forward_hook(self._save("mlp"))
      ]
      return self
  ```
  **Explanation**: Registers a forward hook on the `layer.mlp` module, capturing MLP activations (output of the feed-forward network) after non-linear transformations. Saved as `"mlp"` in `buffers`.

- **Attention Experiment**:
  ```python
  def __enter__(self):
      layer = model.model.layers[self.L]
      self.handles = [
          layer.self_attn.o_proj.register_forward_hook(self._save("attn"))
      ]
      return self
  ```
  **Explanation**: Registers a hook on `layer.self_attn.o_proj`, capturing the attention output after projecting multi-head attention back to the hidden size. Saved as `"attn"` in `buffers`.

- **Residual Stream Experiment**:
  ```python
  def __enter__(self):
      layer = model.model.layers[self.L]
      self.handles = [
          layer.register_forward_hook(self._save("resid"))
      ]
      return self
  ```
  **Explanation**: Registers a hook on the entire `layer` module, capturing the residual stream output (sum of attention, MLP, and prior residual) after the layer’s transformations. Saved as `"resid"` in `buffers`.

#### Activation Retrieval and Collection
- **MLP Experiment** (`ActivationGrabber.pooled` and `collect_layer_features`):
  ```python
  def pooled(self) -> np.ndarray:
      mlp = self._pool(self.buffers["mlp"]).cpu().numpy()
      self.clear()
      return mlp

  def collect_layer_features(layer: int, dataloader: DataLoader, n_examples: int = None):
      feats_mlp, labels = [], []
      with torch.no_grad(), ActivationGrabber(layer) as grabber:
          for batch_txts, batch_domains in tqdm(dataloader, desc=f"Layer {layer}"):
              toks = tokenizer(batch_txts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
              _ = model(**toks)
              mlp = grabber.pooled()
              feats_mlp.append(mlp.astype(np.float32))
              labels.append(np.asarray(batch_domains))
      X_mlp = np.concatenate(feats_mlp, 0)
      y = np.concatenate(labels, 0)
      return X_mlp, y
  ```
  **Explanation**: Retrieves `"mlp"` activations, pools them, and converts to NumPy. In `collect_layer_features`, collects `feats_mlp` into `X_mlp` (`[3000, 4096]` for 3,000 samples, hidden size 4096).

- **Attention Experiment**:
  ```python
  def pooled(self) -> np.ndarray:
      attn = self._pool(self.buffers["attn"]).cpu().numpy()
      self.clear()
      return attn

  def collect_layer_features(layer: int, dataloader: DataLoader, n_examples: int = None):
      feats_attn, labels = [], []
      with torch.no_grad(), ActivationGrabber(layer) as grabber:
          for batch_txts, batch_domains in tqdm(dataloader, desc=f"Layer {layer}"):
              toks = tokenizer(batch_txts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
              _ = model(**toks)
              attn = grabber.pooled()
              feats_attn.append(attn.astype(np.float32))
              labels.append(np.asarray(batch_domains))
      X_attn = np.concatenate(feats_attn, 0)
      y = np.concatenate(labels, 0)
      return X_attn, y
  ```
  **Explanation**: Retrieves `"attn"` activations, pools them, and converts to NumPy. Collects `feats_attn` into `X_attn`, same shape as MLP but containing attention features.

- **Residual Stream Experiment**:
  ```python
  def pooled(self) -> np.ndarray:
      resid = self._pool(self.buffers["resid"]).cpu().numpy()
      self.clear()
      return resid

  def collect_layer_features(layer: int, dataloader: DataLoader, n_examples: int = None):
      feats_resid, labels = [], []
      with torch.no_grad(), ActivationGrabber(layer) as grabber:
          for batch_txts, batch_domains in tqdm(dataloader, desc=f"Layer {layer}"):
              toks = tokenizer(batch_txts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
              _ = model(**toks)
              resid = grabber.pooled()
              feats_resid.append(resid.astype(np.float32))
              labels.append(np.asarray(batch_domains))
      X_resid = np.concatenate(feats_resid, 0)
      y = np.concatenate(labels, 0)
      return X_resid, y
  ```
  **Explanation**: Retrieves `"resid"` activations, pools them, and converts to NumPy. Collects `feats_resid` into `X_resid`, same shape but containing residual stream features.

#### Plot Configuration (`plot_v_bits`)
- **MLP Experiment**:
  ```python
  def plot_v_bits(results):
      layers = [r["layer"] for r in results]
      v_bits = [r["v_bits"] for r in results]
      plt.figure(figsize=(8, 5))
      plt.plot(layers, v_bits, marker='o', label='MLP')
      plt.xlabel('Layer Index')
      plt.ylabel('V-Usable Bits')
      plt.title('V-Usable Bits for MLP Layers')
      plt.grid(True)
      plt.legend()
      plt.show()
  ```
  **Explanation**: Plots v-usable bits with label “MLP” and title “V-Usable Bits for MLP Layers”, reflecting MLP-based features.

- **Attention Experiment**:
  ```python
  def plot_v_bits(results):
      layers = [r["layer"] for r in results]
      v_bits = [r["v_bits"] for r in results]
      plt.figure(figsize=(8, 5))
      plt.plot(layers, v_bits, marker='o', label='Attention')
      plt.xlabel('Layer Index')
      plt.ylabel('V-Usable Bits')
      plt.title('V-Usable Bits for Attention Layers')
      plt.grid(True)
      plt.legend()
      plt.show()
  ```
  **Explanation**: Uses label “Attention” and title “V-Usable Bits for Attention Layers” for attention-based features.

- **Residual Stream Experiment**:
  ```python
  def plot_v_bits(results):
      layers = [r["layer"] for r in results]
      v_bits = [r["v_bits"] for r in results]
      plt.figure(figsize=(8, 5))
      plt.plot(layers, v_bits, marker='o', label='Residual')
      plt.xlabel('Layer Index')
      plt.ylabel('V-Usable Bits')
      plt.title('V-Usable Bits for Residual Stream Layers')
      plt.grid(True)
      plt.legend()
      plt.show()
  ```
  **Explanation**: Uses label “Residual” and title “V-Usable Bits for Residual Stream Layers” for residual stream features.

---

## Results Summary


