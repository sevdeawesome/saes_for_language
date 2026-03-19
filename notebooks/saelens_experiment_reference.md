# SAELens + Gemma Scope 2 + Neuronpedia: Experiment Reference
## Language Feature Identification & Suppression on Gemma 3

---

## 1. Gemma Scope 2: What's Available

Gemma Scope 2 is trained on the **Gemma 3 model family**. SAEs exist for pretrained and instruction-tuned models at sizes: **270M, 1B, 4B, 12B, and 27B**.

### SAE Types Per Repo

- **Single-layer**: `resid_post`, `attn_out`, `mlp_out` — SAEs at 4 layers (25%, 50%, 65%, 85% depth) with various widths/L0s
- **Transcoders**: `transcoder` — skip-transcoders at same 4 layers
- **`_all` variants**: `resid_post_all`, `attn_out_all`, `mlp_out_all`, `transcoder_all` — smaller width/L0 range but **every layer**
- **Multi-layer**: `crosscoder` (weakly causal, 4 concatenated residual stream layers), `clt` (cross-layer transcoders, whole model MLP outputs)

### Which SAE to Use

> Unless you're doing full circuit-style analysis, we recommend using SAEs / transcoders from the layer subset folders, e.g. `resid_post` or `transcoder`.

- **Width**: 16k, 64k, 256k, 1m. Recommended: **64k or 256k**.
- **L0**: "small" (10-20), "medium" (30-60), "large" (60-150). Recommended: **"medium"**.

### Release Names for SAELens Loading

```
# 1B pretrained
gemma-scope-2-1b-pt-resid_post
gemma-scope-2-1b-pt-resid_post_all
gemma-scope-2-1b-pt-attn_out
gemma-scope-2-1b-pt-mlp_out
gemma-scope-2-1b-pt-transcoder

# 27B instruction-tuned
gemma-scope-2-27b-it-resid_post
gemma-scope-2-27b-it-resid_post_all
gemma-scope-2-27b-it-attn_out
gemma-scope-2-27b-it-mlp_out
gemma-scope-2-27b-it-transcoder

# 27B pretrained
gemma-scope-2-27b-pt-resid_post
# ... same pattern
```

### sae_id Format

```
layer_{N}_width_{W}_l0_{size}
```

Examples:
```
layer_12_width_16k_l0_small
layer_12_width_16k_l0_medium
layer_12_width_64k_l0_medium
layer_12_width_256k_l0_large
```

---

## 2. Loading SAEs

### Basic Load

```python
from sae_lens import SAE

sae = SAE.from_pretrained(
    release="gemma-scope-2-1b-pt-resid_post",
    sae_id="layer_12_width_16k_l0_medium",
    device="cuda"
)
```

### Also Returns Config/Sparsity

```python
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2-1b-pt-resid_post",
    sae_id="layer_12_width_16k_l0_small",
)
```

### Key SAE Attributes

```python
print(f"Input dimension (d_in): {sae.cfg.d_in}")
print(f"SAE dimension (d_sae): {sae.cfg.d_sae}")
print(f"Expansion factor: {sae.cfg.d_sae / sae.cfg.d_in}")
print(f"Hook name: {sae.cfg.metadata.hook_name}")
print(f"Model name: {sae.cfg.metadata.model_name}")
print(f"HF Hook name: {sae.cfg.metadata.hf_hook_name}")

# Weights
print(f"Encoder weights shape: {sae.W_enc.shape}")  # (d_in, d_sae)
print(f"Decoder weights shape: {sae.W_dec.shape}")  # (d_sae, d_in)
print(f"Decoder bias shape: {sae.b_dec.shape}")      # (d_in,)
```

---

## 3. Running SAEs Directly (encode / decode / forward)

### Encode — activations → sparse features

```python
import torch
from sae_lens import SAE

sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device="cuda"
)

# activations shape: (batch, seq_len, d_model)
activations = torch.randn(1, 128, 2304, device="cuda")

# feature_acts shape: (batch, seq_len, d_sae)
feature_acts = sae.encode(activations)

# Check which features are active
active_features = (feature_acts > 0).sum(dim=-1)
print(f"Average L0: {active_features.float().mean().item()}")
```

### Decode — sparse features → reconstructed activations

```python
reconstructed = sae.decode(feature_acts)

mse = (activations - reconstructed).pow(2).mean()
print(f"Reconstruction MSE: {mse.item()}")
```

### Forward — full pipeline (encode + decode)

```python
# Equivalent to sae.decode(sae.encode(activations))
reconstructed = sae.forward(activations)

# Or simply:
reconstructed = sae(activations)
```

---

## 4. Using SAEs with Gemma 3 via SAETransformerBridge (Beta)

**This is the path for Gemma 3 models (not supported by HookedTransformer natively).**

> SAETransformerBridge requires TransformerLens v3, which is currently in beta.
> Install it with `pip install transformer-lens>=3.0.0b0`. The API may change in future versions.

### Setup

```python
from sae_lens import SAE
from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

# Load model using TransformerBridge
model = SAETransformerBridge.boot_transformers("google/gemma-3-4b-it", device="cuda")

# Load SAE (Gemma Scope 2 SAEs work with Gemma 3 models)
sae = SAE.from_pretrained(
    release="gemma-scope-2-4b-it-res",
    sae_id="layer_17_width_16k_l0_medium",
    device="cuda"
)
```

### Run with SAEs

```python
# Add SAE permanently
model.add_sae(sae)
logits = model("Hello, world!")
model.reset_saes()

# Or use context manager for temporary attachment
with model.saes(saes=[sae]):
    logits = model("Hello, world!")

# Run with SAEs (temporary, removed after forward pass)
logits = model.run_with_saes("Hello, world!", saes=[sae])

# Run with cache to access SAE activations
logits, cache = model.run_with_cache_with_saes("Hello, world!", saes=[sae])
```

---

## 5. Using SAEs WITHOUT TransformerLens (Pure PyTorch Hooks)

**If SAETransformerBridge doesn't work or you want full control, register hooks yourself.**

> SAEs from SAELens are standard PyTorch modules and can be used with any model or
> framework. The key is extracting activations from your model and passing them to the
> SAE's encode(), decode(), or forward() methods. Also note that the names of hook
> points will be different between TransformerLens and Hugging Face / NNsight.

### Extracting Activations with register_forward_hook

```python
import torch
from transformers import AutoModel, AutoTokenizer
from sae_lens import SAE

# Load Hugging Face model
model = AutoModel.from_pretrained("google/gemma-2-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model.eval()

# Load SAE (trained on Gemma 2 2B residual stream at layer 12)
sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device="cpu"
)

# Storage for activations
activations = {}

def hook_fn(module, input, output):
    # Gemma transformer blocks output a tuple; hidden states are first
    hidden_states = output[0] if isinstance(output, tuple) else output
    activations["layer_12"] = hidden_states.detach()

# Register hook on layer 12
handle = model.layers[12].register_forward_hook(hook_fn)

# Run forward pass
inputs = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    model(**inputs)

# Remove hook
handle.remove()

# Use SAE on extracted activations
layer_12_acts = activations["layer_12"]
feature_acts = sae.encode(layer_12_acts)
reconstructed = sae.decode(feature_acts)

print(f"Input shape: {layer_12_acts.shape}")
print(f"Feature activations shape: {feature_acts.shape}")
print(f"Active features per token: {(feature_acts > 0).sum(dim=-1)}")
```

### Full Example: Finding Top Features

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model.eval()

sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device=device
)

def get_sae_features(text, layer=12):
    """Extract SAE features for a given text."""
    activations = {}

    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        activations["hidden"] = hidden_states.detach()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    feature_acts = sae.encode(activations["hidden"])
    return feature_acts, inputs["input_ids"]

# Analyze a prompt
text = "The capital of France is"
features, tokens = get_sae_features(text)

# Find top active features at the last token
last_token_features = features[0, -1, :]
top_features = torch.topk(last_token_features, k=10)

print(f"Top 10 active features at last token:")
for idx, (feat_idx, value) in enumerate(zip(top_features.indices, top_features.values)):
    print(f"  Feature {feat_idx.item()}: {value.item():.4f}")
```

---

## 6. HookedSAETransformer (For Models It Supports — NOT Gemma 3)

**Warning: Use `from_pretrained_no_processing` to load the model, not `from_pretrained`.
Most SAEs are trained on raw LLM activations, and the default processing in
`from_pretrained` will apply post-processing to the activations, and may break your SAE.**

### Setup

```python
from sae_lens import SAE, HookedSAETransformer

model = HookedSAETransformer.from_pretrained_no_processing("gemma-2-2b", device="cuda")

sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device="cuda"
)
```

### Run with Cache and SAEs

```python
logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

# Access SAE feature activations
sae_acts = cache["blocks.12.hook_resid_post.hook_sae_acts_post"]
print(f"SAE activations shape: {sae_acts.shape}")
```

### Ablate a Feature with Hooks

```python
from functools import partial

def ablate_feature(sae_acts, hook, feature_id):
    sae_acts[:, :, feature_id] = 0.0
    return sae_acts

# Ablate feature 1000 during forward pass
logits = model.run_with_hooks_with_saes(
    tokens,
    saes=[sae],
    fwd_hooks=[
        ("blocks.12.hook_resid_post.hook_sae_acts_post",
         partial(ablate_feature, feature_id=1000))
    ]
)
```

### Add SAEs Persistently

```python
model.add_sae(sae)

# Now standard forward passes include the SAE
logits = model(tokens)
logits, cache = model.run_with_cache(tokens)

# Remove all attached SAEs
model.reset_saes()
```

### Using Error Terms

```python
sae.use_error_term = True
model.add_sae(sae)

# Output is now: SAE(x) + error_term = x (original activation)
logits = model(tokens)

# You can intervene on the error term
logits = model.run_with_hooks(
    tokens,
    fwd_hooks=[
        ("blocks.12.hook_resid_post.hook_sae_error",
         lambda act, hook: torch.zeros_like(act))
    ]
)
```

---

## 7. NNsight Integration

### Extract Features

```python
import torch
from nnsight import LanguageModel
from sae_lens import SAE

model = LanguageModel("google/gemma-2-2b", device_map="auto")

sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device="cuda"
)

prompt = "The Eiffel Tower is located in"

with model.trace(prompt):
    hidden_states = model.model.layers[12].output[0]
    hidden_states_saved = hidden_states.save()

with torch.no_grad():
    features = sae.encode(hidden_states_saved)

print(f"Feature activations shape: {features.shape}")
print(f"Average L0: {(features[:, 1:, :] > 0).sum(dim=-1).float().mean().item():.1f}")
```

### Intervene on SAE Features

```python
def ablate_top_features(hidden_states, sae, k=10):
    """Ablate the top-k active features and return modified activations."""
    features = sae.encode(hidden_states)

    for pos in range(features.shape[1]):
        top_k = torch.topk(features[0, pos], k=k)
        features[0, pos, top_k.indices] = 0.0

    return sae.decode(features)

with model.trace(prompt) as tracer:
    hidden_states = model.model.layers[12].output[0]
    modified = ablate_top_features(hidden_states, sae, k=10)
    model.model.layers[12].output[0][:] = modified
    logits = model.lm_head.output.save()

print(f"Output shape: {logits.shape}")
```

---

## 8. Training SAEs (For Non-TransformerLens Models)

When using HuggingFace models directly for training:

> To use a Huggingface AutoModelForCausalLM, you can specify
> `model_class_name = 'AutoModelForCausalLM'` in the SAE config. Your hook points
> will then need to correspond to the named parameters of the Huggingface model
> rather than the typical TransformerLens hook points. For instance, if you were
> using GPT2 from Huggingface, you would use `hook_name = 'transformer.h.1'`
> rather than `hook_name = 'blocks.1.hook_resid_post'`. Otherwise everything
> should work the same as with TransformerLens models.

---

## 9. Neuronpedia: Feature Search & Exploration

### Python Library

```bash
pip install neuronpedia
```

### Feature Identifiers

A feature on Neuronpedia is identified by three parts:
- **Model ID**: e.g. `gemma-2-2b`
- **Source**: e.g. `3-gemmascope-att-16k` (layer 3, Gemma Scope, attention, 16k width)
- **Index**: string (usually an integer), e.g. `4232`

### Get a Feature via API

```python
from neuronpedia.np_sae_feature import SAEFeature

sae_feature = SAEFeature.get("gemma-2-2b", "3-gemmascope-att-16k", "4232")
print(sae_feature)
```

### JSON API (GET)

```
# Feature page:
https://www.neuronpedia.org/gpt2-small/6-res_scefr-ajt/650

# JSON version (add /api/feature/ after domain):
https://www.neuronpedia.org/api/feature/gpt2-small/6-res_scefr-ajt/650
```

### Steering via Neuronpedia API

```python
import json
import requests

PROMPT = "The most iconic structure on Earth is"
MODEL_ID = "gemma-2b"
FEATURE = {"modelId": "gemma-2b", "layer": "6-res-jb", "index": 10200, "strength": 5}
TEMPERATURE = 0.2
N_TOKENS = 16
FREQ_PENALTY = 1.0
SEED = 16
STRENGTH_MULTIPLIER = 4

url = "https://www.neuronpedia.org/api/steer"
data = {
    "prompt": PROMPT,
    "modelId": MODEL_ID,
    "features": [FEATURE],
    "temperature": TEMPERATURE,
    "n_tokens": N_TOKENS,
    "freq_penalty": FREQ_PENALTY,
    "seed": SEED,
    "strength_multiplier": STRENGTH_MULTIPLIER,
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=headers)
json_response = response.json()
print(json.dumps(json_response, indent=4))
```

### Upload Custom Steering Vectors

```python
from neuronpedia.np_vector import NPVector
import os, json

os.environ["NEURONPEDIA_API_KEY"] = "YOUR_NP_API_KEY"

np_vector = NPVector.new(
    label="dinosaurs",
    model_id="gemma-2-2b-it",
    layer_num=20,
    hook_type="hook_resid_pre",
    vector=your_vector_data,
    default_steer_strength=44,
)

responseJson = np_vector.steer_chat(
    steered_chat_messages=[{"role": "user", "content": "Write a one sentence story."}]
)
print(json.dumps(responseJson, indent=2))
```

### Search Features

Neuronpedia supports two search modes:
1. **Search Explanations** — semantic search over auto-interp labels
2. **Search via Inference** — runs custom text through model, returns top activating features

Both available via the web UI at neuronpedia.org and via API.

### SAELens ↔ Neuronpedia Integration

```python
from sae_lens.analysis.neuronpedia_integration import (
    open_neuronpedia_feature_dashboard,
    get_neuronpedia_quick_list,
    get_neuronpedia_feature,
)

# Open single feature dashboard in browser
open_neuronpedia_feature_dashboard(sae, feature_idx=1234)

# Open quick list of multiple features
get_neuronpedia_quick_list(sae, feature_indices=[100, 200, 300])

# Fetch feature data programmatically (no browser)
feature_data = get_neuronpedia_feature(sae, feature_idx=1234)

# The neuronpedia_id is auto-set on SAE.from_pretrained()
print(sae.cfg.metadata.neuronpedia_id)
```

---

## 10. Hook Name Mapping

When using HuggingFace models directly (not TransformerLens), hook point names differ:

| TransformerLens | HuggingFace |
|---|---|
| `blocks.1.hook_resid_post` | `model.layers[1]` (register_forward_hook) |
| `blocks.1.hook_mlp_out` | `model.layers[1].mlp` |
| `blocks.1.attn.hook_z` | `model.layers[1].self_attn` |

For Gemma 3 specifically, the HF model structure is:
```
model.layers[N]           # → residual stream after layer N
model.layers[N].mlp       # → MLP output
model.layers[N].self_attn # → attention output
```

When extracting activations via hook:
```python
def hook_fn(module, input, output):
    # Gemma transformer blocks output a tuple; hidden states are first
    hidden_states = output[0] if isinstance(output, tuple) else output
    activations["layer_N"] = hidden_states.detach()
```

---

## Appendix: Installation

```bash
pip install sae-lens
pip install transformer-lens>=3.0.0b0  # for SAETransformerBridge (beta)
pip install neuronpedia                 # for Neuronpedia Python API
```
