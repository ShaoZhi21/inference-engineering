# Week 1 Guide — Transformers + nanoGPT

## Goal

Build enough of GPT from scratch that the transformer forward pass is no longer magic.

By the end of Week 1, you should be able to explain this path with tensor shapes:

```text
token ids -> token embeddings -> positional embeddings -> self-attention -> MLP -> logits -> loss
```

Do not optimize yet. Do not implement KV cache yet. Week 1 is the clean forward pass.

## Primary Resource

Karpathy, "Let's build GPT: from scratch, in code, spelled out."

Use the video as the main guide. Code along manually. The goal is not to clone a finished implementation; the goal is to write each piece after you understand why it exists.

## Does the nanoGPT Repo Have a Guide?

Yes. The repo README at `https://github.com/karpathy/nanoGPT` includes:

- install command
- Shakespeare character-level quick start
- CPU command with smaller hyperparameters
- Apple Silicon `mps` note
- sampling command
- finetuning and GPT-2 reproduction notes
- troubleshooting

But use it carefully. The repo is described as a simple, fast training/finetuning repo, not primarily an educational walkthrough. Its `train.py` and `model.py` are readable, but they are already the cleaned-up version. For Week 1, first build the small educational version from the video, then run the official repo as a comparison.

## Folder Plan

Recommended layout:

```text
nanogpt/
  README.md
  input.txt
  bigram.py
  gpt.py
  notes.md
```

Keep this separate from `mnist-classifier`.

## Setup

From the repo root:

```bash
mkdir -p nanogpt
cd nanogpt
```

Use the existing top-level virtual environment if it already has PyTorch:

```bash
source ../.venv/bin/activate
python -c "import torch; print(torch.__version__)"
```

If PyTorch is missing, install the small Week 1 dependencies:

```bash
pip install torch numpy
```

For the official nanoGPT repo later, the README lists:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

For the educational Week 1 build, start with only `torch` and `numpy`.

## Dataset

Use a tiny text file first. Karpathy uses Tiny Shakespeare.

The file should be:

```text
nanogpt/input.txt
```

At this stage the dataset is just raw text. You will build a character vocabulary:

```text
chars -> integer ids -> tensors
```

## Milestones

### 1. Character Tokenizer

Write:

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join([itos[i] for i in ids])
```

Understand:

```text
text character -> integer token id
integer token id -> text character
```

### 2. Train/Validation Split

Convert all text into one long tensor:

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

Then split:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

### 3. Batching

Implement `get_batch(split)`.

Track shapes:

```text
x: (B, T)
y: (B, T)
```

Where:

```text
B = batch size
T = block size / context length
```

For language modeling:

```text
x = current tokens
y = next tokens
```

### 4. Bigram Model

Start with the simplest model:

```text
token id -> logits for next token
```

This proves the training loop works before attention enters the picture.

Track:

```text
idx:    (B, T)
logits: (B, T, vocab_size)
loss:   scalar
```

### 5. Self-Attention Head

Implement a single attention head.

Track:

```text
x:   (B, T, C)
q:   (B, T, head_size)
k:   (B, T, head_size)
v:   (B, T, head_size)
wei: (B, T, T)
out: (B, T, head_size)
```

Key idea:

```text
attention weights decide which earlier tokens each token reads from
```

Use a causal mask so position `t` cannot see future positions.

### 6. Multi-Head Attention

Run several heads in parallel and concatenate:

```text
head outputs -> concat -> projection
```

Why:

```text
different heads can learn different token relationships
```

### 7. Feed-Forward MLP

Add the per-token MLP after attention:

```text
Linear -> ReLU/GELU -> Linear
```

Attention mixes information across time. The MLP transforms each token position independently.

### 8. Transformer Block

Combine:

```text
self-attention
residual connection
layer norm
MLP
residual connection
layer norm
```

Know the purpose:

- residuals help gradients and preserve signal
- layer norm stabilizes activations
- attention mixes across tokens
- MLP computes per-token features

### 9. GPT Model

Final structure:

```text
token embedding
positional embedding
N transformer blocks
final layer norm
linear language-model head
```

Track final shape:

```text
logits: (B, T, vocab_size)
```

### 10. Generation

Autoregressive generation loop:

```text
take current context
predict next-token logits
sample one next token
append it
repeat
```

For Week 1, generation can be inefficient. It is okay to recompute the full context every step. Week 2 is where you add KV cache and measure the speedup.

## Shape Checklist

Keep this in `notes.md` and fill it in while coding:

```text
B =
T =
C =
vocab_size =
head_size =
n_head =
n_layer =

idx shape =
token embedding shape =
position embedding shape =
q/k/v shape =
attention score shape =
attention output shape =
logits shape =
loss shape =
```

If you cannot fill this out, slow down before adding more code.

## Official nanoGPT Repo Pass

After your educational version runs, use the official repo to see the production-shaped version.

Suggested flow:

```bash
git clone https://github.com/karpathy/nanoGPT.git external/nanoGPT
cd external/nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm
python data/shakespeare_char/prepare.py
```

CPU run from the repo README:

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Apple Silicon run:

```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Sample:

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

Use `--device=mps` for sampling too if you trained with MPS.

## What To Compare Against Your Version

Read these files in the official repo:

```text
model.py
train.py
config/train_shakespeare_char.py
data/shakespeare_char/prepare.py
sample.py
```

Compare:

- where embeddings are created
- how causal attention is masked
- where layer norm is placed
- how the training loop estimates validation loss
- how generation crops context to `block_size`
- how config overrides work

## Deliverable

By the end of Week 1, produce:

```text
nanogpt/gpt.py
nanogpt/notes.md
```

`gpt.py` should:

- load `input.txt`
- tokenize characters
- train a tiny GPT
- print train/validation loss
- generate sample text

`notes.md` should include:

- tensor shape table
- short explanation of causal masking
- short explanation of attention vs MLP
- one paragraph comparing your educational version to official nanoGPT

## Stop Point

Stop Week 1 when:

- your tiny GPT trains
- generated text becomes more structured than random characters
- you can explain every major tensor shape
- you can identify the attention block in both your code and nanoGPT `model.py`

Do not implement KV cache yet. That is Week 2.
