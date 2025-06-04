<!-- Custom vLLM Fork with Logits Support -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Custom Fork of vLLM with Logits Output Support
</h3>

---

## 🔍 Problem Statement

The default vLLM implementation only returns **log-probabilities**, computed via softmax on model logits. However, these are often **overly sharp** and unsuitable for applications requiring:

- Smoother or calibrated probability distributions
- Custom sparse representation generation
- Interpretable or rule-based token selection
- Retrieval-based weighting using raw logits

Moreover, once softmax is applied, it is mathematically impossible to reconstruct the original logits.

## ✅ Solution

We patched the original vLLM repository to optionally **return raw logits** from the LLM server, enabling downstream applications to apply **custom logits-to-probability strategies**.

### 🔧 Implementation Summary

We modified the following section in `vllm/model_executor/layers/sampler.py`:

```python
# In Sampler.forward(), before _build_sampler_output is called:
if logits is not None:
    logits_cpu = logits.cpu()
    logits_cpu_len = len(logits_cpu)
    single_logprob_len = logits_cpu_len // len(prompt_logprobs)
    for prompt_id, prompt_logprob in enumerate(prompt_logprobs):
        if prompt_logprob is not None:
            d = prompt_logprob[0]
            k = list(d.keys())[0]
            logprob = d[k].logprob
            sharded_logits_cpu = logits_cpu[prompt_id * single_logprob_len : (prompt_id + 1) * single_logprob_len, :]
            logprob_column = torch.full((len(sharded_logits_cpu), 1), logprob, dtype=sharded_logits_cpu.dtype, device='cpu')
            logits_w_logprob = torch.cat([sharded_logits_cpu, logprob_column], dim=1)
            d[k].logprob = logits_w_logprob
    del logits
    torch.cuda.empty_cache()
```

This hack appends the full logits (optionally with a reference logprob) to the dictionary that will be returned by the API response.

---

## 📦 Installation

Clone this modified fork:

```bash
git clone https://github.com/hyukkyukang/vllm-logits-output.git
cd vllm-logits-output
```

Then install the package via pip:

```bash
pip install .
```

You may need to uninstall the original vLLM version first:

```bash
pip uninstall vllm
```

---

## 🚀 Usage

Once installed, your LLM server (e.g., FastAPI endpoint) can now access `logits` from `prompt_logprobs` as part of the server's response. You can extract, post-process, or convert them using:

- Temperature-scaled softmax
- Entropy regularization
- Top-k filtering and renormalization
- Custom masking or rule-based filters

This enables more flexible and interpretable downstream behavior.

---

## ❗Disclaimer

This is a fork of the official [vLLM project](https://github.com/vllm-project/vllm) and not officially supported by the core team. The patch may require maintenance as upstream changes occur.

For core features, documentation, and updates, please refer to the [main repository](https://github.com/vllm-project/vllm).

---

## 📬 Contact

For questions or collaboration regarding this fork, please contact [@hyukkyukang](https://github.com/hyukkyukang).

---

> This fork enables custom control over logits-to-probability conversion while maintaining the high-performance LLM inference pipeline of vLLM.
