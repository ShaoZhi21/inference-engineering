# 2-Month Ramp Plan — Fireworks Applied ML Engineer

**Goal:** convert interview-level concepts (quantization, KV cache, paged attention, LoRA, SFT/DPO/RLHF, evals) into hands-on, under-the-hood skill that maps to the AMLE day job: build PoCs, deploy end-to-end apps, enable new models, optimize performance, fine-tune.

**Assumed budget:** ~8–10 hrs/week → ~1 hr × 4 weekdays (reading + small hands-on) + one weekend block (2–4 hrs, project work). Sustainable around a 9–5 and a relationship. Dial up/down per week.

**Principle:** no toy ML *as the destination*. Every project past Week 0 is something you could show a Fireworks customer or rebuild on the platform. The agentic RAG over 10-Ks is the foundation — the capstone extends *that* onto your own served, fine-tuned, benchmarked stack.

**Compute:** small dense models (4–8B) fit one GPU. Options: Colab Pro / RunPod / Lambda / Modal for a few hours at a time; Ollama or llama.cpp for local poking; Fireworks + Baseten free tiers for the "serve it like the platform does" parts. Don't try to self-host the big MoE models (Kimi K2.6, DeepSeek V4) — use APIs for those, self-host the small ones.

---

## Theme 0 — Starters: PyTorch fundamentals + a number classifier

Why: you can't reason about an inference bottleneck or read vLLM/TRL/nanoGPT code if the basic PyTorch loop is unfamiliar. This is the on-ramp everything else stands on. It's the *one* place toy ML is the right call — get a working win, build muscle memory, move on.

**Resources (core)**
- PyTorch official *Learn the Basics* tutorial — tensors, `Dataset`/`DataLoader`, `nn.Module`, autograd, the training loop, devices/dtypes.
- The classic MNIST digit-classifier walkthrough (PyTorch docs or any standard tutorial).

**Project — "MNIST digit classifier" (the standard intro project)**
1. Load MNIST via `torchvision.datasets`.
2. Build a small net (`nn.Linear` layers, or a tiny CNN if you want).
3. Write the training loop yourself: forward → loss → `backward()` → optimizer step.
4. Evaluate test accuracy; plot a few predictions and misclassifications.

Deliverable: a working classifier and a training loop you wrote by hand. This exact `Dataset → DataLoader → nn.Module → train/eval` pattern is reused by nanoGPT, TRL fine-tuning, and vLLM internals — Week 0 is where it becomes muscle memory.

**Stretch (optional, role-relevant):** after it trains, cast the weights fp32 → int8, re-measure accuracy and on-disk model size. ~5 lines, and it's your first concrete taste of quantization — the lever you'll spend all of Week 4 on, on a model small enough to understand completely.

### Optional depth — going under the hood (micrograd)

Not scheduled, not a prerequisite. In the classifier you *call* `loss.backward()` as a black box; micrograd re-implements that one method in ~100 lines of pure Python so you understand *what it actually does*.

- Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd* (Zero-to-Hero, Lesson 1).

The natural slot is right after the classifier works — you've used `.backward()`, now build it. The one place it pays off later is Theme 2's memory reasoning (KV cache, activation memory), which is all about *what the backward pass keeps around*. Half a day. Do it if a week runs light; skip it freely otherwise — every later theme works without it.

---

## Theme 1 — Transformers + PyTorch

Why: you can't reason about an attention optimization if the forward pass is a black box. This unblocks the inference work.

**Resources (core)**
- Karpathy, *Neural Networks: Zero to Hero* — specifically "Let's build GPT from scratch" and "Let's reproduce GPT-2 (nanoGPT)". The canonical hands-on build. Free, with code.
- *Attention Is All You Need* (Vaswani et al., 2017) — read it **after** the Karpathy build so the diagrams map to code you've written.
- Jay Alammar, *The Illustrated Transformer* — intuition pass before/alongside.

**Resources (optional)**
- 3Blue1Brown attention/transformer videos — geometric intuition.
- *The Annotated Transformer* (Harvard NLP) — line-by-line PyTorch of the paper.

**Project — "nanoGPT + KV cache"**
1. Build nanoGPT from scratch following Karpathy; train a char-level model.
2. Then modify it yourself: implement a **KV cache** for autoregressive decoding and measure the tokens/sec difference. This makes the single most important inference concept concrete.
3. Stretch: swap in a different attention variant (e.g. grouped-query attention) and note the memory/quality trade-off.

Deliverable: a repo + a short note on what KV caching actually bought you and why.

---

## Theme 2 — Inference & serving (the core of this role)

Why: Fireworks *is* an inference company. "Performance optimizations" and "new model enablements" are literal responsibilities. This is where you should spend the most time.

**Resources (core)**
- Baseten, *Inference Engineering* (Philip Kiely) — free PDF. Read Ch 2 (Models/attention bottlenecks), Ch 3 (Hardware/GPU spec sheets), Ch 4 (Software: CUDA → PyTorch → vLLM/SGLang/TensorRT-LLM/Dynamo), Ch 5 (Techniques: quantization, spec decoding, KV reuse, parallelism, disaggregation). Skim Ch 6–7.
- vLLM docs + **PagedAttention** paper (Kwon et al., SOSP 2023).
- SGLang docs + **RadixAttention** / SGLang paper (prefix-cache reuse).
- Lilian Weng, *Large Transformer Model Inference Optimization* — production-grade survey blog.

**Resources (optional)**
- FlashAttention v1/v2 (Dao et al.) — at least the blog + abstracts; understand "IO-aware."
- Speculative decoding (Leviathan et al., 2023).
- Quantization: GPTQ and AWQ papers, or the HF quantization guide.

**Project — "vLLM vs SGLang benchmark harness"**
1. Serve a small open model (e.g. a Qwen3 8B dense or Gemma small) under **both vLLM and SGLang**.
2. Build a load-test harness measuring **TTFT, p50/p99 latency, and throughput (tok/s)**.
3. Sweep the levers you've only read about: batch size / continuous batching, **quantization** (FP16 vs INT8 vs INT4-AWQ), KV-cache / prefix-cache settings, tensor parallelism (if you have 2 GPUs).
4. Write a one-page benchmark report: which lever moved what, and the cost-per-million-tokens implication.

This is the single most role-relevant artifact you can walk in with. Stretch: add a tiny model gateway that routes between two models by request type.

---

## Theme 3 — Fine-tuning (SFT, LoRA/QLoRA, DPO)

Why: the JD lists SFT and RLHF/RFT as preferred, and **multi-LoRA serving at inference time is a core Fireworks value prop** — not just training. You should be able to do the whole loop: tune → serve the adapter → measure the delta.

**Resources (core)**
- HuggingFace **TRL** docs (`SFTTrainer`, `DPOTrainer`) + PEFT docs (LoRA/QLoRA).
- Sebastian Raschka — LoRA writing + *Build a Large Language Model (From Scratch)* for the training mechanics.
- **LoRA** (Hu et al., 2021) and **QLoRA** (Dettmers et al., 2023).
- Fireworks fine-tuning + LoRA + multi-LoRA serving docs — know the request/response shapes cold.

**Resources (optional)**
- InstructGPT (Ouyang et al., 2022) for the RLHF mental model; **DPO** (Rafailov et al., 2023) for the preference-optimization method Fireworks supports.

**Project — "tune → serve → eval delta"**
1. LoRA/QLoRA SFT a small open model on a focused domain dataset (single GPU, TRL).
2. Run a **DPO** pass on a small preference set.
3. **Serve the adapter** via multi-LoRA (vLLM LoRA support, or Fireworks) — the part most people skip.
4. Run a small binary-scored eval harness (Hamel-style methodology) before vs after. Report the delta and the serving cost of hot-swapping adapters.

This mirrors the exact AMLE loop and reuses your eval strengths.

---

## Theme 4 — SWE / Systems (DDIA + production serving)

Why: PoCs and customer apps live or die on the glue around the model. DDIA gives you the distributed-systems vocabulary; the project gives you the production shape.

**Resources (core)**
- **DDIA 2nd ed** (Kleppmann & Riccomini, Feb 2026). Don't read all 672 pages — target: Trade-offs in Data Systems Architecture, Nonfunctional Requirements, Storage & Retrieval, Replication, Partitioning, Consistency, and the Batch/Stream chapters. Read as a slow background thread, ~20–30 pp/week.

**Project — "production inference gateway"**
Build the layer a real PoC needs in front of your Theme-2 deployment:
- FastAPI service with request queuing + batching/debounce (workers pull, queues never push).
- **Redis** cache for repeated prompts (write-through: DB/source first, then Redis).
- Per-user **rate limiting** (key includes `user_id`; token bucket vs semaphore for the two distinct jobs).
- Basic observability: latency, throughput, and token accounting.

This is the "Application Build" + "PoC" responsibility, built from DDIA primitives you'll recognize.

---

## Papers — tiered (the "several papers" you wanted)

**Read (core):** Attention Is All You Need (2017) · PagedAttention/vLLM (2023) · LoRA (2021) · QLoRA (2023) · InstructGPT (2022) · DPO (2023).

**Skim (optional):** FlashAttention v1/v2 · Speculative decoding · GPTQ/AWQ · a scaling-laws paper (Chinchilla) · an MoE paper (Mixtral or similar) — useful because most of today's top open models (Qwen, DeepSeek, Kimi) are MoE, and you'll be explaining their serving characteristics to customers.

---

## 8-Week Schedule

One primary hands-on theme at a time; DDIA runs lightly in the background throughout so you're never doing two GPU projects at once. Week 0 is the on-ramp — keep it short and time-boxed so it doesn't eat into Weeks 4 and 6.

| Week | Primary focus | Hands-on milestone | Background read |
|------|---------------|--------------------|-----------------|
| 0 | PyTorch fundamentals | **MNIST digit classifier** (standard intro) — train loop by hand; *optional:* int8 quantize + measure | — |
| 1 | Transformers + PyTorch | Karpathy build → working nanoGPT | DDIA: architecture trade-offs |
| 2 | Transformers + PyTorch | Add KV cache, measure speedup; read AIAYN | DDIA: nonfunctional requirements |
| 3 | Inference & serving | Serve a small model on vLLM; baseline benchmarks | Baseten Ch 2–3; DDIA: storage/retrieval |
| 4 | Inference & serving | Add SGLang; sweep quant/batch/KV; **write benchmark report** | Baseten Ch 4–5; DDIA: partitioning |
| 5 | Fine-tuning | LoRA/QLoRA SFT a small model (TRL) | LoRA + QLoRA papers; DDIA: replication |
| 6 | Fine-tuning | DPO pass → serve adapter (multi-LoRA) → eval delta | DPO paper; Fireworks fine-tuning docs |
| 7 | SWE / serving layer | FastAPI gateway + Redis cache + rate limit + metrics | DDIA: consistency |
| 8 | **Capstone** | Stitch it together (see below) | DDIA: batch/stream; wrap up |

> Note: Week 0 is the starter on-ramp and isn't counted against the 8-week core — it can overlap your existing routine at lower intensity. The optional micrograd "under the hood" detour (see Theme 0) slots between Week 0 and Week 1 only if a week runs light.

**Capstone (Week 8):** take your 10-K agentic RAG and put it on *your own* stack — self-served small model behind your Theme-7 gateway, a fine-tuned router/intent model from Theme 3 (you already use Qwen3-8B for routing), benchmarked and cost-modeled from Theme 2, evaluated with your existing harness. That's a complete customer-shaped PoC and a strong story for day one.

---

## Keeping it sane

- Pick **one** project per fortnight; reading fills the weekday hours, the build is the weekend.
- Keep a running gaps doc — log every "I don't actually understand X" as you hit it; that list is your real curriculum.
- It's fine to drop the optional papers, the micrograd detour, and the capstone stretch if a week gets busy. The benchmark report (Wk 4) and the tune→serve→eval loop (Wk 6) are the two things worth protecting.
- If you finish early, the highest-leverage extras are: (a) the micrograd under-the-hood detour, or (b) reproducing one Fireworks/Baseten blog benchmark yourself.
