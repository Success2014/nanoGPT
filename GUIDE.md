# Understanding nanoGPT: A Step-by-Step Guide

This document summarizes how to read and run this repository. It covers the core files (`train.py`, `model.py`), data flow, next-token prediction, training loop, and inference — based on hands-on experiments on Mac M4 Max, single/multi-GPU H20, and cluster notebooks.

For launch commands and recorded benchmark numbers, see [README.md](README.md) sections **How to conduct experiment** and **Experiment results**.

---

## 1. What is this repo?

**nanoGPT** is Karpathy's minimal GPT training codebase. The README states the essence:

> Core files: `train.py` (~300 lines, training loop) and `model.py` (~300 lines, GPT model). Optionally load GPT-2 weights from OpenAI.

### End-to-end pipeline

```
Raw text
    ↓  data/*/prepare.py
train.bin / val.bin   (token id sequences, binary)
    ↓  train.py
checkpoint (out_dir/ckpt.pt)
    ↓  sample.py
Generated text
```

| Stage | Files | Role |
|-------|-------|------|
| Data prep | `data/*/prepare.py` | Text → token ids → `.bin` |
| Training | `train.py` + `model.py` | Read data, forward, loss, backward, update |
| Config | `config/*.py`, `configurator.py` | Hyperparameters without editing core code |
| Inference | `sample.py` | Load checkpoint, autoregressive generation |

**What to focus on:**

| Priority | File | Content |
|----------|------|---------|
| ★★★ | `train.py` | Training loop, data loading, optimizer, DDP |
| ★★★ | `model.py` | GPT architecture, forward, loss, generate |
| ★★ | `data/*/prepare.py` | Data format |
| ★ | `config/*.py` | Experiment settings |
| ★ | `sample.py` | Sampling / inference |

---

## 2. Data preparation

Training input is **not** raw text — it is a long 1D array of **token ids** (integers). `prepare.py` converts text once; `train.py` only reads `.bin` files during training.

### Two tokenization styles

| Dataset | Script | Tokenizer | Use case |
|---------|--------|-----------|----------|
| `data/shakespeare_char/` | char-level | 65 chars → ids 0–64 | Quick demo (~3 min) |
| `data/shakespeare/`, `data/openwebtext/` | BPE | tiktoken GPT-2 (~50257 vocab) | Finetune / GPT-2 reproduction |

### What `prepare.py` does (shakespeare_char example)

1. Load `input.txt`
2. Split train (90%) / val (10%)
3. Map each character to an integer (`stoi` / `itos`)
4. Write `train.bin`, `val.bin` as flat `uint16` arrays
5. Optionally save `meta.pkl` (vocab for decode)

### How `train.py` reads data: `get_batch`

```python
# train.py (simplified)
ix = torch.randint(len(data) - block_size, (batch_size,))
x = data[i : i+block_size]      # input
y = data[i+1 : i+1+block_size]   # target = x shifted right by 1
```

Example (`block_size=5`):

```
data:  [T, h, e,  , c, a, t, ...]

x:     [T, h, e,  , c]
y:     [h, e,  , c, a]
```

This is **next token prediction** at the data layer: given tokens at positions `0..T-1`, predict the token at each next position.

---

## 3. Next token prediction in code

GPT-2 is trained as **next token prediction**: given preceding tokens, predict the next one.

### Three layers

**1. Data — `(x, y)` shift**

- `y[t] = x[t+1]`
- One forward pass trains `block_size` next-token tasks in parallel

**2. Model — logits at each position**

```python
# model.py forward (training)
tok_emb = wte(idx)           # (B, T, n_embd)
pos_emb = wpe(pos)           # position info
x = dropout(tok_emb + pos_emb)
for block in transformer.h:
    x = block(x)
x = ln_f(x)
logits = lm_head(x)          # (B, T, vocab_size)
loss = cross_entropy(logits, targets)
```

**3. Architecture — causal attention**

- `is_causal=True` in attention: position `i` may only attend to `0..i`
- Prevents "seeing the future" during training

### Training vs inference

| | Training | Inference (`generate`) |
|--|----------|------------------------|
| Input | Full sequence `X (B, T)` | Prompt + tokens generated so far |
| Output | Logits at **every** position | Logits at **last** position only |
| Next step | Compare to `Y`, backprop | Sample one token, append, repeat |

Autoregressive loop (inference):

```
prompt: "The cat"
  → predict " sat"  → "The cat sat"
  → predict " on"   → "The cat sat on"
  → ...
```

---

## 4. Tensor shapes: B and T

`idx` has shape **`(B, T)`**:

| Dim | Name | Meaning | Typical (shakespeare_char) |
|-----|------|---------|---------------------------|
| 0 | **B** | Batch size — how many sequences in parallel | 64 |
| 1 | **T** | Sequence length (time steps) | 256 (`block_size`) |

After embedding:

```
idx:     (B, T)              token ids
hidden:  (B, T, n_embd)       e.g. (64, 256, 384)
logits:  (B, T, vocab_size)   e.g. (64, 256, 65)
```

`idx[b, t]` = token id for sample `b` at position `t`.

---

## 5. Model structure (`model.py`)

GPT is built from four building blocks:

```
GPTConfig (hyperparameters)
    ↓
CausalSelfAttention + MLP  →  Block (× n_layer)
    ↓
Embedding + n_layer × Block + lm_head  →  GPT
```

### GPTConfig (GPT-2 124M defaults)

| Param | Default | Meaning |
|-------|---------|---------|
| `vocab_size` | 50304 | Token vocabulary size |
| `block_size` | 1024 | Max context length |
| `n_layer` | 12 | Transformer blocks |
| `n_head` | 12 | Attention heads |
| `n_embd` | 768 | Hidden dimension |

### Architecture diagram

```
idx (B, T)
    ├─ wte (token embedding)
    ├─ wpe (position embedding)  → add, dropout
    ↓
Block × n_layer:
    x = x + Attn(LayerNorm(x))    # causal self-attention
    x = x + MLP(LayerNorm(x))     # feed-forward
    ↓
ln_f → lm_head → logits (B, T, vocab_size)
```

### Block internals

- **CausalSelfAttention**: Q, K, V from one linear; causal mask; optional Flash Attention (`scaled_dot_product_attention`)
- **MLP**: `768 → 3072 → GELU → 768`
- **Pre-LN + residual**: LayerNorm before sublayers; `x + sublayer(x)`

### Weight tying

```python
self.transformer.wte.weight = self.lm_head.weight
```

Input embedding and output projection share weights — fewer parameters, often better quality.

### `forward(idx, targets=None)`

- **Training**: `targets` given → full-sequence logits + cross-entropy loss
- **Inference**: `targets=None` → only last-position logits (cheaper)

---

## 6. Training loop (`train.py`)

### Structure

```
① Load config (defaults + config file + CLI)
② Init device / DDP / seeds
③ Create model (scratch | resume | gpt2 pretrained)
④ Optimizer, optional torch.compile, DDP wrap
⑤ while True:
      set learning rate
      periodic eval + checkpoint
      forward → backward → optimizer.step()
      until iter_num > max_iters
```

### Config system

`train.py` defines globals at the top; `configurator.py` overrides from:

```
train.py defaults  <  config/foo.py  <  --key=value
```

Example:

```bash
python train.py config/train_shakespeare_char.py --batch_size=32 --compile=False
```

### Key mechanisms

**Gradient accumulation**

Simulate larger batch when memory is limited:

```
micro_step 0..N-1: forward → loss/N → backward (accumulate grads)
then: optimizer.step()
```

`gradient_accumulation_steps` must be **divisible by total GPU count** in DDP.

**Learning rate schedule** (`get_lr`)

1. Linear warmup (`warmup_iters`)
2. Cosine decay to `min_lr`
3. Flat at `min_lr` after `lr_decay_iters`

**Eval and checkpoint**

- `estimate_loss()`: average loss over `eval_iters` batches on train/val
- Save `ckpt.pt`: model, optimizer, `iter_num`, `best_val_loss`, config

### One training iteration (data flow)

```
train.bin
  → get_batch() → X, Y  (B, T)
  → model(X, Y) → logits, loss
  → loss.backward()
  → clip_grad_norm → optimizer.step()
```

---

## 7. Distributed training (DDP)

nanoGPT uses **Distributed Data Parallel (DDP)** for multi-GPU — **not** model parallelism or pipeline parallelism (unnecessary at GPT-2 124M scale).

### Single node, 8 GPUs

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py \
  --gradient_accumulation_steps=8 \
  --compile=False --dtype=float32 --disable_flash=True
```

- One process per GPU
- Each GPU holds a **full copy** of the model
- Each processes a different micro-batch; gradients **AllReduce** across GPUs

### Multi-node (e.g. 4 nodes × 8 GPUs = 32)

Run one command per node; only `--node_rank` changes. `gradient_accumulation_steps` must divide **32**.

### Reading logs

| Field | Meaning |
|-------|---------|
| `tokens per iteration` | All GPUs combined tokens per optimizer step |
| `time XXms` | Wall time for one step (master rank) — **not** 1/8 of single-GPU time |
| `mfu XX%` | **Single-GPU** Model FLOPs Utilization vs A100 peak — not sum over 8 GPUs |

**Speedup** shows up in **tokens/second**, not necessarily in per-iteration wall time.

Example (shakespeare_char, stable H20 settings):

| Setup | tokens/iter | time/iter | Throughput |
|-------|-------------|-----------|------------|
| 1× GPU | 16,384 | ~36 ms | ~455K tok/s |
| 8× GPU | 131,072 | ~37 ms | ~3.5M tok/s |

Same `max_iters=5000` on 8 GPUs ⇒ **~8× more total tokens** ⇒ stronger overfitting on tiny datasets (low train loss, **worse** val loss). For fair comparison: `--max_iters=625` with `--gradient_accumulation_steps=8`.

---

## 8. Inference and finetuning

### `sample.py`

```
Load model (ckpt or gpt2*) → encode prompt → model.generate() → decode
```

- `temperature`, `top_k`: control randomness
- `device=auto`: CUDA → MPS (Mac) → CPU

### Finetuning

Same as training; differences are config-only:

| | Pretrain | Finetune |
|--|----------|----------|
| `init_from` | `scratch` | `gpt2`, `gpt2-xl`, ... |
| `learning_rate` | ~6e-4 | ~3e-5 |
| `max_iters` | 600k | tens |
| Data | OpenWebText | Small domain text |

---

## 9. Device notes (MPS, H20)

### Apple Silicon (MPS)

- `device_utils.py`: `device=auto` picks MPS on Mac
- Training: `--device=mps --compile=False` recommended

### NVIDIA H20 stability

Some stacks crash with `Floating point exception` after step-0 eval when using default `bfloat16` + Flash Attention backward.

**Stable recipe:**

```bash
python train.py config/train_shakespeare_char.py \
  --compile=False \
  --dtype=float32 \
  --disable_flash=True
```

Code fixes (when synced):

- `disable_flash=True`: manual attention instead of SDPA
- bf16: disable fused AdamW
- Flash path: apply attention dropout after SDPA, not inside `dropout_p`

---

## 10. Mental model checklist

Before moving to the next topic, you should be able to answer:

1. **What files matter?** → `train.py`, `model.py`, data `prepare.py`, `config/`, `sample.py`
2. **What is in `.bin`?** → 1D `uint16` token ids; `(x,y)` with `y` shifted by 1
3. **What does GPT predict?** → Next token at each position; causal mask enforces past-only context
4. **What are B and T?** → Batch size and sequence length
5. **What does one training step do?** → `get_batch` → `forward` → `loss` → `backward` → `step`
6. **Multi-GPU?** → DDP; match `gradient_accumulation_steps` to GPU count; compare tokens/s
7. **MFU?** → Single-GPU utilization estimate; ~10% on small models is normal

---

## 11. Suggested learning path

1. Read this guide + skim `README.md` quick start
2. Run `prepare.py` + single-GPU `train.py` on `shakespeare_char`
3. Read `get_batch` and `GPT.forward` side by side
4. Read `Block` and `CausalSelfAttention` in `model.py`
5. Trace one iteration in `train.py`'s `while True` loop
6. Run `sample.py` on `out-shakespeare-char`
7. Try 8-GPU DDP with fair `max_iters` / grad accum settings
8. Scale to OpenWebText + `config/train_gpt2.py` when ready

For deeper theory, Karpathy's [Zero To Hero GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) matches this codebase closely.

---

## 12. File reference

| File | Lines (approx) | Purpose |
|------|----------------|---------|
| `train.py` | ~350 | Config, DDP, data loader, train loop, checkpoint |
| `model.py` | ~335 | GPT, Block, attention, MLP, `generate`, `from_pretrained` |
| `configurator.py` | ~48 | Override globals from config file / CLI |
| `device_utils.py` | ~40 | auto/MPS/CUDA device and dtype helpers |
| `sample.py` | ~90 | Load checkpoint, generate text |
| `config/train_shakespeare_char.py` | ~38 | Small char-level demo |
| `config/train_gpt2.py` | ~26 | GPT-2 124M on OpenWebText |
| `config/finetune_shakespeare.py` | ~26 | Finetune gpt2-xl on Shakespeare |
