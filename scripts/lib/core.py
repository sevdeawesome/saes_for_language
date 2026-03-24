"""Core utilities for language-specific SAE feature experiments on Gemma 3.
Comments written by Claude Opus 4.6. Most code comes from the Google Gemma Scope 2 tutorial
(with some minor modifications).

Design notes:
- Gemma 3-4B-IT is Gemma3ForConditionalGeneration (multimodal wrapper), so text layers
  live at model.model.language_model.layers[N], not model.model.layers[N] like 1B.
- SAEs are loaded via SAELens. PT SAEs work fine on IT models (same features emerge).
- Suppression uses projection removal: subtract strength * activation * decoder_direction
  from the residual stream. This removes the feature's contribution without breaking others.
- Steering uses Google's norm-scaled approach from their Gemma Scope 2 tutorial:
  resid += coeff * ||resid|| * W_dec[feat]. The norm scaling is critical — without it,
  the decoder vector (~unit norm) is negligible vs residual stream norms (~hundreds).
  The hook also handles KV-caching: steer only last token on prefill, every token after.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


def load_model_and_sae(model_id, sae_release, sae_id):
    """Load HF model + tokenizer + SAELens SAE. Returns (model, tokenizer, sae)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16)
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device="cuda")
    return model, tokenizer, sae


def get_layer(model, n):
    """Get layer N, handling both 1B (model.model.layers) and 4B+ (model.model.language_model.layers)."""
    inner = model.model
    return inner.language_model.layers[n] if hasattr(inner, "language_model") else inner.layers[n]


def get_mean_sae_activations(model, tokenizer, sae, layer, texts):
    """Run texts through model, hook residual stream at `layer`, return mean SAE activation per feature.

    We skip the BOS token (position 0) since it carries no language signal.
    Returns a 1D tensor of shape (d_sae,) with mean activation across all tokens and texts.
    """
    total = torch.zeros(sae.cfg.d_sae, device="cuda")
    n_tokens = 0
    for text in texts:
        cache = {}
        def hook(mod, inp, out, cache=cache):
            cache["r"] = out[0].detach() if isinstance(out, tuple) else out.detach()
        h = get_layer(model, layer).register_forward_hook(hook)
        try:
            model(**tokenizer(text, return_tensors="pt").to(model.device))
        finally:
            h.remove()
        resid = cache["r"][:, 1:, :].to("cuda")
        total += sae.encode(resid.float()).sum(dim=(0, 1))
        n_tokens += resid.shape[1]
    return total / n_tokens


def make_suppression_hook(sae, features, strength):
    """Projection suppression: subtract strength * feat_activation * decoder_direction.

    This surgically removes a feature's contribution from the residual stream.
    Works because SAE features are approximately linear: the feature's effect on the
    residual stream is act * W_dec[feat], so subtracting a multiple of that removes it.
    """
    def hook(mod, inputs, outputs):
        resid = outputs[0] if isinstance(outputs, tuple) else outputs
        r = resid.to(torch.float32).to("cuda")
        fa = sae.encode(r)
        for fi in features:
            r = r - strength * (fa[:, :, fi:fi+1] @ sae.W_dec[fi:fi+1, :])
        out = r.to(resid.dtype).to(resid.device)
        return (out,) + outputs[1:] if isinstance(outputs, tuple) else out
    return hook


def make_steering_hook(sae, features, coeff):
    """Norm-scaled steering (Google's approach from Gemma Scope 2 tutorial).

    Adds coeff * ||residual|| * decoder_direction to the residual stream for each feature.
    The norm scaling is essential: without it, the perturbation is negligible because
    W_dec rows are ~unit norm while residual stream vectors have norm ~100-1000.

    KV-cache aware: during generate(), the first forward pass sees all prompt tokens
    (shape [1, seq_len, d]), subsequent passes see one new token (shape [1, 1, d]).
    We steer only the last token on prefill, and every token during cached generation.

    Args:
        features: single feature index (int) or list of feature indices
    """
    # Normalize to list
    if isinstance(features, int):
        features = [features]
    # Sum decoder directions for all features
    dec_vec = sum(sae.W_dec[fi] for fi in features)
    def hook(mod, inputs, outputs):
        output = outputs[0]
        v = dec_vec.to(output.device).to(output.dtype)
        if output.shape[1] == 1:
            norm = torch.norm(output, dim=-1, keepdim=True)
            output = output + coeff * norm * v
        else:
            norm = torch.norm(output[0, -1:], dim=-1, keepdim=True)
            output[0, -1:] = output[0, -1:] + coeff * norm * v
        return (output,) + outputs[1:] if isinstance(outputs, tuple) else output
    return hook


def generate(model, tokenizer, prompt, hook_fn=None, layer=None, max_new_tokens=80):
    """Generate a response, optionally with a hook on the specified layer.

    Applies chat template, strips the template wrapper from output.
    """
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)
    h = get_layer(model, layer).register_forward_hook(hook_fn) if hook_fn else None
    try:
        out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        if h:
            h.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=False)
    if "<start_of_turn>model" in text:
        text = text.split("<start_of_turn>model")[-1].strip()
    if "<end_of_turn>" in text:
        text = text.split("<end_of_turn>")[0].strip()
    return text
