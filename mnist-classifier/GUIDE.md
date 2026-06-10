# MNIST Digit Classifier — Build Guide

> Your first hands-on project. The goal is **fluency, not a finished artifact you copied.** By the end you should be able to write a PyTorch training loop from memory and explain every line. Use AI as little as possible — when stuck, re-read the relevant section here, then the official docs, *then* ask.

---

## 1. What "MNIST" even is (long form)

**MNIST** = **M**odified **N**ational **I**nstitute of **S**tandards and **T**echnology database.

The original data came from NIST, the US government standards lab, which had collected handwritten digits from two groups: Census Bureau employees and American high-school students. In 1998, Yann LeCun, Corinna Cortes, and Christopher Burges *modified* it — re-mixing the two groups, normalizing the images, and centering each digit — to make a clean benchmark. That "modified" remix is MNIST.

What's actually in it:

- **70,000 grayscale images** of handwritten digits 0–9.
  - **60,000** for training, **10,000** for testing (the split is fixed and standard — everyone uses the same test set, which is why results are comparable across decades of papers).
- Each image is **28 × 28 pixels**, single channel (grayscale, not RGB). So one image is a grid of 784 numbers, each a pixel intensity from 0 (black) to 255 (white).
- Each image has a **label**: the integer 0–9 it represents.

Why it's *the* intro project:

- **Small enough to be instant.** 28×28 is tiny; you can train a decent model on a laptop CPU in a couple of minutes. No GPU required.
- **Hard enough to be real.** Handwriting varies — your model genuinely has to *learn*, not memorize. A trivial model gets ~90%; a good one gets ~98–99%. That gap teaches you something.
- **Completely understandable.** 10 classes, visible inputs. When your model misclassifies a 4 as a 9, you can look at the image and often see why. You cannot do that with an 8B language model.
- **It's the "hello world" everyone shares.** The training-loop shape you write here — `Dataset → DataLoader → model → loss → backward → step` — is the *exact same shape* used by nanoGPT, by HuggingFace TRL fine-tuning, and inside vLLM. Learn it once, here, where it's cheap.

The task itself is **multi-class image classification**: given the 784 pixel values, output which of 10 digits it is.

---

## 2. What you're building

A program that:

1. Downloads MNIST.
2. Defines a small neural network mapping 784 pixels → 10 class scores.
3. Trains it on the 60k training images.
4. Reports accuracy on the 10k test images.
5. Shows a handful of predictions visually.

**Definition of done:**

- [ ] Test accuracy **≥ 95%** (a basic net hits this easily; don't chase 99% on day one).
- [ ] You wrote the **training loop by hand** — not a `Trainer` wrapper, not copied wholesale.
- [ ] You can explain, out loud, what each of these does: `DataLoader`, `nn.Module.forward`, the loss function, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`.
- [ ] You logged a **gaps doc** (section 10) as you went.

**Stretch (optional, do only after the above works):** quantize the trained weights to int8 and re-measure accuracy + file size (section 9).

---

## 3. Environment setup

```bash
# from this folder (mnist-classifier/)
python -m venv .venv
source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
```

Sanity check it imported:

```bash
python -c "import torch; print(torch.__version__)"
```

You'll write your code in `src/`. Start with one file, `src/train.py`. (A notebook is fine too if you prefer — but writing a plain `.py` you can run end-to-end is closer to how real code ships.)

---

## 4. Conceptual primer — the pieces you'll assemble

Read this once before coding. Each piece is a noun you'll turn into code in section 5.

**Tensor.** PyTorch's array type — like a NumPy array but it can live on a GPU and it tracks operations for autograd. An MNIST image becomes a tensor of shape `[1, 28, 28]` (channels, height, width). A *batch* of 64 images is `[64, 1, 28, 28]`.

**Dataset.** An object that knows how to give you one `(image, label)` pair by index. `torchvision.datasets.MNIST` is a ready-made one — it handles the download and the file format.

**Transform.** A function applied to each image as it's loaded. You'll use `ToTensor()` (converts the 0–255 image to a `[0,1]` float tensor) and usually `Normalize(mean, std)` (re-centers pixel values so training is more stable). For MNIST the canonical numbers are mean `0.1307`, std `0.3081` — those are the precomputed average and spread of the training set's pixels.

**DataLoader.** Wraps a Dataset and hands you *batches* (e.g. 64 images at a time) instead of one image, shuffles the training data each epoch, and can load in parallel. You train on batches because it's faster and the gradient is less noisy than one-image-at-a-time.

**Model (`nn.Module`).** Your network. You subclass `nn.Module`, define layers in `__init__`, and define how data flows through them in `forward`. For MNIST, the simplest working model: flatten the 784 pixels → one or two `Linear` layers with a `ReLU` between → 10 output numbers (one score per digit).

**Logits.** The 10 raw output numbers. They're *not* probabilities yet — they're unnormalized scores. The highest one is the model's prediction.

**Loss function.** Measures how wrong the predictions are versus the true labels. For multi-class classification you use **cross-entropy** (`nn.CrossEntropyLoss`). Conveniently, it takes raw logits directly (it applies softmax internally), so your model should output logits, not probabilities.

**Optimizer.** The thing that adjusts the model's weights to reduce the loss. Start with `Adam` or `SGD`. It needs two things: the model's parameters and a *learning rate* (how big a step to take).

**Autograd / `backward()`.** When you call `loss.backward()`, PyTorch walks backwards through every operation that produced the loss and computes the gradient (the slope) for each weight — i.e. "which direction, and how much, does each weight need to move to lower the loss." `optimizer.step()` then applies that move. This is the black box you'll later open in the optional micrograd detour.

**Epoch.** One full pass over the training set. You'll do a handful (3–5 is plenty for MNIST).

---

## 5. Build plan — milestones

Do these in order. Each milestone should *run* before you move on. The snippets below are deliberately skeletal — fill in the `TODO`s yourself; that's the point.

### Milestone 1 — Load the data and look at it

Before any model, prove you can get data and *see* it. This catches setup problems early.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_set = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# TODO: wrap each in a DataLoader. Train loader: batch_size=64, shuffle=True.
#       Test loader:  batch_size=1000, shuffle=False.

# TODO: grab one batch (next(iter(train_loader))) and print the shapes of
#       images and labels. Confirm images are [64, 1, 28, 28] and labels [64].
```

**Stop and verify:** print one batch's shapes. Optionally use `matplotlib` to show a single image with its label — seeing a real digit render is a good morale check and confirms `ToTensor` worked.

> Gotcha: the first run downloads ~10 MB into `data/`. That folder is gitignored — don't commit it.

> **Two ways to get the data.** The snippet above uses `torchvision` (the classic path). Once your whole pipeline works end-to-end, come back and redo *just* the data loading via **Hugging Face `datasets`** (section 6) — same model, same training loop, but the ecosystem you'll use for every model and dataset in the rest of the ramp.

### Milestone 2 — Define the model

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define layers.
        #   - nn.Flatten() to turn [B,1,28,28] into [B,784]
        #   - nn.Linear(784, 128), then nn.ReLU(), then nn.Linear(128, 10)
        #   (You may use nn.Sequential to hold them, or keep them as fields.)

    def forward(self, x):
        # TODO: pass x through your layers and return the 10 logits per image.
        raise NotImplementedError
```

**Stop and verify:** instantiate `Net()`, feed it one batch, confirm the output shape is `[64, 10]`. If it errors on shapes, you've found your first real lesson — read the error, check what each layer expects.

### Milestone 3 — The training loop (the heart of it)

This is the part you must write yourself and understand cold. The four-line dance inside the batch loop is the thing you'll reuse for the rest of the ramp plan.

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"   # MNIST is fine on CPU too
model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # ---- the core four steps ----
        optimizer.zero_grad()          # 1. clear old gradients
        logits = model(images)         # 2. forward pass -> predictions
        loss = loss_fn(logits, labels) # 3. how wrong are we?
        loss.backward()                # 4a. compute gradients
        optimizer.step()               # 4b. update weights
        # ------------------------------

    # TODO: print the epoch number and the last loss value so you can watch it fall.
```

**Understand before continuing — quiz yourself:**

- Why `zero_grad()` *first*? (PyTorch *accumulates* gradients by default; if you don't clear them, this batch's gradient piles on top of the last one's. Forgetting this is the single most common beginner bug.)
- What shape does `logits` have, and what shape does `CrossEntropyLoss` expect for labels? (Logits `[B, 10]`, labels `[B]` of class indices — *not* one-hot.)
- What would happen if you removed `loss.backward()`? (Gradients stay zero, `step()` moves nothing, loss never drops.)

**Stop and verify:** run it. The printed loss should *decrease* across epochs. If it doesn't, that's your first debugging exercise — don't paper over it, find out why (wrong lr? forgot `zero_grad`? labels on wrong device?).

### Milestone 4 — Evaluate on the test set

Training loss falling isn't the goal — *generalization* is. Measure accuracy on data the model never trained on.

```python
model.eval()
correct, total = 0, 0
with torch.no_grad():                  # don't track gradients — faster, less memory
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)   # the index of the highest logit = predicted digit
        # TODO: count how many preds == labels; accumulate correct and total.

# TODO: print accuracy = correct / total. Aim for >= 95%.
```

**Understand:** why `model.eval()` and `torch.no_grad()`? `eval()` switches layers like dropout/batchnorm into inference mode (harmless here, essential habit later). `no_grad()` tells PyTorch not to build the autograd graph — you're not training, so you don't need gradients; it's faster and uses less memory. This distinction (train vs. inference mode) is *exactly* the train/serve split you'll care about in the inference themes.

### Milestone 5 — Look at predictions

Use `matplotlib` to show ~6 test images with their predicted vs. true labels. Find a misclassification and look at it — often you'll agree it's a genuinely ambiguous scrawl. This is your first taste of *error analysis*, a skill you'll lean on hard in the eval-harness work later.

### Milestone 6 — Save the model

```python
torch.save(model.state_dict(), "mnist_net.pt")   # gitignored
```

`state_dict()` is just a dictionary of the learned weight tensors. Saving/loading weights separately from code is the standard PyTorch pattern — and understanding that a model *is* a bag of tensors is the foundation for the quantization stretch.

---

## 6. The Hugging Face path — self-downloading datasets & models

Why this matters: `torchvision.datasets.MNIST` is the classic on-ramp, but **Hugging Face is the ecosystem the rest of this ramp plan lives in.** Every model you serve in Theme 2 and fine-tune in Theme 3 comes from the HF Hub; most datasets too. Learning to pull a dataset and a model from the Hub *now*, on something trivial, means it's muscle memory when the stakes are higher.

Do this as a **second pass** after your torchvision version works: same model, same training loop, different data source. The whole point is to feel that the model and loop *don't change* — only the plumbing does.

### 6.1 Control where things download

By default HF caches everything in `~/.cache/huggingface` — a hidden global cache. While learning, you want to *see* your downloads, so point them at this folder:

- Per call: pass `cache_dir="data/hf"`.
- Globally for a shell session: `export HF_HOME="$PWD/data/hf"` before running.

Either way it lands under `data/` (already gitignored). Knowing where weights actually live on disk is the same instinct you'll need when you're managing model storage on a serving box.

### 6.2 Loading MNIST from the `datasets` library

```python
from datasets import load_dataset

ds = load_dataset("ylecun/mnist", cache_dir="data/hf")
print(ds)                 # DatasetDict: 'train' (60k) and 'test' (10k)
print(ds["train"][0])     # {'image': <PIL.Image ...>, 'label': 5}
```

Two differences from torchvision to notice:

- Images arrive as **PIL Image** objects in an `image` column, labels as ints in a `label` column. *You* convert images to tensors — there's no built-in transform pipeline.
- You get one `DatasetDict` (splits are keys) rather than two separate objects.

Wiring it into the DataLoader/loop you already wrote:

```python
from torchvision import transforms

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# TODO: turn ds["train"] into something a DataLoader can batch. Two clean options:
#   (a) ds.with_transform(fn)  — fn maps a batch dict -> tensors, applied lazily.
#   (b) a tiny class(torch.utils.data.Dataset) whose __getitem__ returns
#       (to_tensor(row["image"]), row["label"]).
# Then build a DataLoader exactly like Milestone 1. The model + training loop
# are UNCHANGED — prove that to yourself.
```

The lesson worth internalizing: **the data source changed, the training loop didn't.** That separation between data plumbing and model is exactly what the DDIA reading and the inference-gateway project (Theme 4) are about.

### 6.3 Self-downloading a model from the Hub

You won't classify MNIST with a pretrained Hub model (there isn't a standard one) — but practicing the *download mechanics* now pays off directly in Theme 3, where you'll pull real LLMs. Two equivalent tools:

CLI — download a whole repo into a folder you choose:

```bash
# newer CLI:  hf download <org/model> --local-dir ./models/<name>
# older alias (still works): huggingface-cli download <org/model> --local-dir ./models/<name>

# a tiny model that's quick to pull, just to practice the mechanics:
hf download prajjwal1/bert-tiny --local-dir ./models/bert-tiny
```

Python — the same thing programmatically:

```python
from huggingface_hub import snapshot_download
snapshot_download("prajjwal1/bert-tiny", local_dir="models/bert-tiny")
```

Now **open `./models/bert-tiny` and look at what a model actually is** — this is the payoff:

- `config.json` — the architecture description (layer sizes, counts). Read it; it's just JSON.
- `model.safetensors` — the weights. This is the same `state_dict` (a bag of named tensors) you saved in Milestone 6, in HF's safe serialization format.
- `tokenizer.json` / `vocab.txt` — how text becomes token IDs (irrelevant for MNIST, central later).

Seeing that a "model" is *config + a bag of weight tensors + (maybe) a tokenizer* demystifies the entire Hub. Later you'll load these with `from_pretrained("models/bert-tiny")`, pointing at the local folder instead of re-downloading.

### 6.4 Optional — push your trained MNIST model to the Hub

If you want the full round-trip: `huggingface-cli login` (paste a token from your HF account settings), then `upload_folder` or `push_to_hub` to publish your `mnist_net.pt`. Skippable, but it closes the loop on "download → train → publish" and shows you how your future fine-tuned adapters will be shared.

---

## 7. Suggested file layout

Keep it simple. One file is genuinely fine for this:

```
mnist-classifier/
  GUIDE.md            <- this file
  requirements.txt
  src/
    train.py          <- everything: data, model, train, eval, save
  data/               <- auto-downloaded datasets (torchvision + HF cache), gitignored
  models/             <- self-downloaded HF models (section 6), gitignored
```

If you want to stretch your structure muscles, split into `data.py` (loaders), `model.py` (the `Net`), `train.py` (loop + eval). Optional — don't over-engineer a 100-line project.

---

## 8. Knobs to play with (after it works)

Change **one** at a time and watch what happens to loss and test accuracy. This builds intuition for the "sweep the levers" work in Theme 2.

- **Learning rate** (`1e-2`, `1e-3`, `1e-4`) — too high and loss explodes/oscillates; too low and it crawls.
- **Batch size** (`32`, `64`, `256`) — affects speed and gradient noise.
- **Epochs** (`1` vs `10`) — watch for diminishing returns.
- **Model width/depth** (hidden size `128` → `256`; add a second hidden layer) — more capacity, slower, marginal gains on MNIST.
- **Optimizer** (`Adam` vs plain `SGD` with `lr=0.1`) — feel the difference.

Log each experiment's accuracy. That table *is* a mini benchmark report — the same artifact you'll produce at scale in Week 4.

---

## 9. Stretch — int8 quantization (the role-relevant bonus)

Only after everything above works. This is your first real inference experiment.

The idea: your weights are 32-bit floats (`float32`). Most of that precision is wasted — you can often store them as 8-bit integers (`int8`), making the model **~4× smaller**, usually with little accuracy loss. This is *the* core lever of the entire serving industry, and MNIST lets you see it on a model you fully understand.

The simplest path is PyTorch **dynamic quantization**, which quantizes the `Linear` layers:

```python
import torch

quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# TODO: re-run your Milestone-4 eval loop on `quantized`. How much accuracy did you lose?
# TODO: torch.save both models' state_dicts and compare file sizes on disk.
```

Write 3–4 sentences: how much accuracy did int8 cost, and how much size did it save? That paragraph is the seed of the cost/quality reasoning the whole job revolves around.

---

## 10. Keep a gaps doc

Make a file `src/GAPS.md` (or a note wherever you like) and log every "wait, I don't actually get X" the moment you hit it. Examples of the kind of thing worth logging:

- "Why does CrossEntropyLoss want logits, not softmax probabilities?"
- "What exactly is `dim=1` in `argmax`?"
- "Why normalize with 0.1307 / 0.3081 specifically?"

Don't necessarily resolve them all now. This list is your real curriculum — it tells you what to read next and what to ask about. Carrying a gaps doc through all 8 weeks is the single highest-leverage habit in the whole plan.

---

## 11. Common pitfalls (check here before asking AI)

| Symptom | Likely cause |
|---|---|
| Loss doesn't decrease / is `nan` | Learning rate too high; or forgot `optimizer.zero_grad()`; or forgot `loss.backward()`. |
| Shape error in `forward` | Didn't `Flatten` the image, or `Linear` in/out sizes don't match the data. |
| Loss falls but test accuracy is bad | Evaluating in `train()` mode, or comparing against wrong labels, or a device mismatch. |
| `RuntimeError: expected ... cuda ... cpu` | An image/label tensor and the model are on different devices — `.to(device)` everything. |
| CrossEntropyLoss errors on labels | You one-hot-encoded the labels; it wants raw integer class indices `[B]`. |
| Accuracy stuck at ~10% | Model isn't learning at all — gradients not flowing (check the four core steps are all present and in order). |

---

## 12. Glossary (quick reference)

- **Logit** — a raw, unnormalized output score (one per class).
- **Softmax** — turns logits into probabilities that sum to 1 (CrossEntropyLoss does this for you).
- **Cross-entropy** — the loss for classification; high when the model is confidently wrong.
- **Gradient** — the slope of the loss w.r.t. a weight; which way to nudge it.
- **Epoch** — one full pass over the training data.
- **Batch** — a small group of examples processed together in one step.
- **`state_dict`** — the dictionary of a model's learned weight tensors.
- **Quantization** — storing weights/activations in fewer bits (e.g. int8) to shrink and speed up a model.
- **Hugging Face Hub** — the public registry where models, datasets, and tokenizers are hosted; you pull from it with `datasets`, `huggingface_hub`, and `transformers`.
- **safetensors** — Hugging Face's safe, fast file format for storing a model's weight tensors (a `state_dict` on disk).
- **`DatasetDict`** — what `load_dataset` returns: a dict whose keys are splits (`train`, `test`) and values are datasets.

---

### When you're done

You should be able to delete `mnist_net.pt`, re-run `train.py` from scratch, and get a ≥95% model in a couple of minutes — and narrate what every block is doing while it runs. That's Week 0 complete and the foundation for nanoGPT laid.
