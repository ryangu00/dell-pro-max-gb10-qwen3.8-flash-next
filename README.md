# Qwen3.8-Flash-Next on two Dell Pro Max with GB10 — from a 50K safety line (first recipe, 2026-08-31) to a 1M-context production engine (2026-09-20)

> A field report on dual-node deployment of a new-architecture GDN+QSA model. Quality is strong (it beat DeepSeek V4 Flash in our paired evals), prefill is blazing fast —
> but on the **first recipe we tried, this stack had a long-context kill line that nominal specs will never show you**. Half of this book is a deployment guide for that first recipe; the other half is a methodology for finding the kill line on your own stack. Part 2 moves to a community cluster recipe where that kill line no longer reproduces.

> **Status (2026-09-20):** the 2026-08-31 findings in Part 1 are **historical** — they were properties of that first recipe, not of the model.
> With a community cluster recipe (Part 2) the long-context kill line no longer reproduces, and **this model became our production model on 2026-09-20**.

## Hardware and Versions

| Item | Spec/Version |
|---|---|
| Machine | Dell Pro Max with GB10 ×2 (200GbE direct link, same topology as the DSV4F book in this series) |
| Weights (Part 1 recipe) | 126GiB ×2 shards; revision `7b71922` belongs to the Part 1 recipe's weight repository ([MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)). After downloading, **pin that revision and run the recipe's `check-weights.sh` to verify shard integrity** |
| Architecture | GDN (Gated DeltaNet) + QSA hybrid attention — **long-context behavior differs from full-attention models; this is the core of this book** |

## Part 1 — First Recipe (2026-08-31)

> ⚠️ The numbers below were measured on the **first recipe** we tried (the MiaAI-Lab dual-Spark recipe). They describe that recipe on this stack, **not** the model. Part 2 revisits the same model on a different recipe and the picture changes completely.

| Metric | Value (first recipe) |
|---|---|
| Prefill | 2321 tok/s (~30K single-shot prompt, single-seat, TP2 dual-node — the GDN architecture dividend) |
| Quality | Beat DeepSeek V4 Flash in paired evals; agentic semantic understanding was significantly stronger on our paired samples (the other model misread "if it rains, switch to an online meeting" as "reschedule the online meeting" and miscalculated relative dates; this model did not) |
| Nominal context | 1M |
| **Measured safety line (first recipe)** | **≤50K (K = thousand tokens); from ~95K the worker process is killed** (vLLM sm_121 dual-node stack) |

### Finding and Measuring the Long-Context Kill Line (on the first recipe)

Nominal `max_model_len=1M`, the config boots fine, everything works under light traffic — then a single ~95K prompt kills the worker process. The worker process disappeared without an OOM message or a timeout in the logs we kept; we did not capture exit codes, so we cannot say more than "killed". The failure was 100% reproducible with the same prompt. After bisection probing: ≤50K is safe, 50K–95K is unstable, ≥95K always dies.

**Methodology (applies to bringing up any new-architecture model)**:
1. Don't trust `max_model_len` — that's the model-side nominal figure, not a promise about "your engine + your kernel stack + your hardware". Kernel paths for GDN/linear-attention architectures on new hardware (sm_121) are far less mature than full-attention paths.
2. Before going live, run a **length-ladder measurement**: 8K→32K→50K→95K→144K→200K. Minimal reproducible method:
   - Generator: pad with repeated paragraphs to the target token count, and insert one unique fact (a needle, e.g. "the warehouse key is in the third drawer") at the beginning/middle/end of the text;
   - At each rung, call `/v1/chat/completions` asking for the needle; pass = the answer contains that unique fact;
   - Record: HTTP status / worker process liveness (`pgrep` before vs. after) / latency. Worker gone = killed; we did not capture exit codes, so we do not distinguish native crash from OOM killer or external termination.
3. Once you find the kill line, **hard-code the safe ceiling in the service config** (rather than noting it in docs and hoping callers behave).
4. Someone else's green is not your green: the same model on a different engine / single node / different quantization may not have this problem at all — we've seen people run the same model on long inputs on other stacks without issue. The kill line belongs to the *combination*, not the model. (Part 2 is the proof of that last sentence: the kill line below is a property of the recipe, and it moved when we changed recipes.)

### First-recipe deployment essentials

```bash
# The first recipe ships start.sh/stop.sh/check-weights.sh; the flow:
# 1. Download weights (126GiB×2, be patient) → check-weights.sh to verify shard integrity
# 2. In .env, likewise zero out the *_HOST_IP placeholders (same as pitfall #1 in this series' DSV4F book)
# 3. Set max_model_len to the measured safety line (we set ≤50K); do not copy the nominal 1M
```

### How we ended up using the first recipe

Verdict (2026-08-31): **standby**, not production — it won on quality, but the 50K hard limit is a structural risk for our long-context workloads (codebase injection, long documents). Weights and config stayed fully in place, switchable with one command (zero-deletion config principle).

## Part 2 — What Changed (2026-09-18 to 2026-09-20)

We moved off the first recipe onto a **community cluster recipe**: a vLLM `b12x` MoE backend, TP2 over RoCE, MTP-4 speculative decoding, fp8 KV cache. On this recipe the long-context kill line from Part 1 did **not** reproduce, and the model cleared our full private eval bank. The 2026-08-31 verdict is reversed: this model is now production.

### Long context at 200K and then 1M

- The 200K-token tier of our private eval bank ran **two full rounds with zero crashes**.
- Context was then extended to **1M with YaRN factor 4**. Caveat: the MTP draft model needs `max_model_len` passed inside `--speculative-config`, otherwise it stays capped at 262144.
- KV pool: **3.44M tokens**.
- 950K-token cold end-to-end runs: **three runs of 856–860 s (about 14.3 minutes each), all correct**.
- Needle content recall: **9/9 at each of 400K / 700K / 950K**.
- Single-stream decode: **52 tok/s**.
- Six-stream decode: **82.1 tok/s aggregate** (aggregate tok/s of six concurrent streams, measured with async scheduling off). With async scheduling on the same measurement is 89.4, so turning async off costs about −8%.
- Cold prefill throughput: **3034 tok/s**, measured on a ~2.3K-token prompt. The 950K figure above is an end-to-end wall for a 24-token answer, not a prefill-throughput number.

### A runaway repetition loop, isolated

- A **~1/40 runaway repetition loop** was isolated. The identified trigger is the interaction between vLLM async scheduling and MTP speculative decoding; turning async scheduling off removed every case in the 120-question isolation set (3/120 → 0/120).
- The flag is `--no-async-scheduling`.
- Cost of the flag: six-stream throughput **−8%** (see the measurement above).

### The agentic gap narrowed in thinking mode

- In **non-thinking mode** an agentic gap remained: agentic-if **81.7** vs **95.0** for DeepSeek V4 Flash.
- In **thinking mode** with the official sampling, the gap narrowed to **1.7 points (93.3 vs 95.0 three-run medians)**, within one question's worth of noise on this 30-question category.
- The **full thinking-mode verdict passed all 11 categories** of our private eval bank. (The questions themselves are not published.)

### Native 262K vs 1M

- Native 262K remains the **better form on the categories we measured**: kbqa **+5.0** (percentage points, in thinking mode), long-coding wall **0.62×** (the wall-clock ratio native / 1M, in thinking mode).
- 1M is therefore a **per-workload choice**, not a blanket upgrade.

### Production since 2026-09-20

- **Production since 2026-09-20**, with **thinking effort low on the fast tier** and **medium on the quality tier**.

## When to Pick This Setup

✅ High-quality creative/agentic workloads — and now also long-context workloads on the Part 2 recipe (configured to 1M and validated up to 950K-token prompts — needle content recall 9/9 at 400K, 700K and 950K — with zero crashes across two full 200K rounds)
✅ Short-context workloads that feed on prefill speed and semantic understanding
❌ Workloads that need 1M on the first recipe (Part 1) — use the Part 2 recipe instead

## Where to Read the Details

| Focus | Repo |
|---|---|
| 1M-context extension (YaRN factor 4, `--speculative-config` `max_model_len`, KV pool, needle recall, cold prefill timing) | [dell-pro-max-gb10-qwen3.8-flash-next-1m-context](https://github.com/ryangu00/dell-pro-max-gb10-qwen3.8-flash-next-1m-context) |
| The async-scheduling × MTP runaway repetition loop and the `--no-async-scheduling` fix | [dell-pro-max-gb10-vllm-mtp-async-runaway](https://github.com/ryangu00/dell-pro-max-gb10-vllm-mtp-async-runaway) |
| The agentic gap, thinking mode, official sampling, and the 11-category eval-bank verdict | [dell-pro-max-gb10-qwen3.8-flash-next-agentic-thinking](https://github.com/ryangu00/dell-pro-max-gb10-qwen3.8-flash-next-agentic-thinking) |
| The engine A/B (first recipe vs community cluster recipe) | [dell-pro-max-gb10-qwen3.8-flash-next-engine-ab](https://github.com/ryangu00/dell-pro-max-gb10-qwen3.8-flash-next-engine-ab) |

---
*RyanAI Lab · All numbers measured on our resident environment. Updated 2026-09. Issues welcome.*
