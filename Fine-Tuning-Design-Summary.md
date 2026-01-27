## Links

* **Notebook:** https://colab.research.google.com/drive/1Au53Sk_-OUWKgllHCBTV0rd_PBkrl7GN?usp=drive_link
* **Repo:** https://github.com/MartinZuo1/RapidFire-Competition
* **Screenshots (link):** https://github.com/MartinZuo1/RapidFire-Competition/tree/main/auxiliary/screenshots


# Fine-Tuning Experiment Summary

## 1) What you tried (2–4 sentences)

I fine-tuned a small language model to perform **abstractive dialogue summarization**: given a messenger-style chat, generate a short third-person summary of what the participants discussed. The target user is anyone who wants quick digestion of chat threads (e.g., support chats or team discussions). I used the **SAMSum Corpus**, which contains messenger-like conversations written by linguists and paired with human-written summaries; the dataset includes informal language, typos, and speaker names, which makes it realistic for chat summarization.

## 2) What “good” looks like (success criteria)

Good means the model produces **concise, third-person summaries** that capture the key facts from the dialogue while staying on-task and on-format (summary only, not copying the entire dialogue). I measure improvement primarily with **ROUGE-1 / ROUGE-2 / ROUGE-L** on a held-out eval split (plus average generated length as a sanity check for verbosity).

## 3) Setup (bullet list)

* **Base model(s):** `gpt2` (124M) causal LM with LoRA adapters (PEFT).
* **Dataset(s) / domain (size, format, any filtering/cleanup):**

  * **SAMSum** dialogue summarization dataset. Fields: `dialogue`, `summary`, `id`.
  * Official splits: **train 14,732 / val 818 / test 819**. ([Hugging Face][1])
  * For free Colab constraints, I used a **small shuffled subset** for rapid iteration (e.g., 64 train, 10 eval) and kept the pipeline reproducible with a fixed seed.
* **Train/Eval split:** train = `train` split; eval = `val` split.
* **Prompt / formatting approach (1 line: what the model sees as input/output):**

  * Input: an instruction + `Dialogue: ...`
  * Output: `Summary: ...` (the reference summary as the completion)
* **Training budget (fixed across runs):**

  * `max_steps = 32`, batch size 2, grad accumulation 2 (effective batch size ≈ 4 examples/step), eval every 4 steps, fp16 + gradient checkpointing.

## 4) Experiment dimensions (what you varied + why)

* **Knob 1: Prompt variant (2)** — *why:* summarization quality is sensitive to instruction wording and delimiters; two prompts test whether the model benefits from explicit constraints (e.g., “1–2 sentences”, “concise”).
* **Knob 2: LoRA rank r (2: r=8 vs r=16)** — *why:* rank controls adapter capacity vs compute/memory; higher rank can model more task-specific behavior but may overfit on small data.
* **Knob 3: Learning rate (2: 5e-4 vs 2e-4)** — *why:* LR strongly affects stability and convergence speed in LoRA SFT; lower LR is often more stable, higher LR may learn faster.

## 5) Configs compared (keep it short)

Naming convention (so your results table stays readable):

* **Baselines (zero-shot, no fine-tuning):**

  * **Baseline V1:** GPT-2 zero-shot with **Prompt V1**
  * **Baseline V2:** GPT-2 zero-shot with **Prompt V2**
* **Config A:** Prompt V1 + LoRA r=8 + LR=5e-4
* **Config B:** Prompt V1 + LoRA r=8 + LR=2e-4
* **Config C:** Prompt V1 + LoRA r=16 + LR=5e-4
* **Config D:** Prompt V1 + LoRA r=16 + LR=2e-4
* **Config E:** Prompt V2 + LoRA r=8 + LR=5e-4
* **Config F:** Prompt V2 + LoRA r=8 + LR=2e-4
* **Config G:** Prompt V2 + LoRA r=16 + LR=5e-4
* **Config H:** Prompt V2 + LoRA r=16 + LR=2e-4

(These eight correspond exactly to **2×2×2**: prompt × rank × LR.)


## 6) Results (tiny table)

Notes:

* Fine-tuned metrics below are the **raw “Value” at step 32** from TensorBoard (not smoothed).
* Baselines were computed separately via **zero-shot generation** using the same evaluation code and then ROUGE against the eval references.
* Because the eval split was very small (e.g., 10 examples), these scores are **noisy** and best used for relative comparison within this sweep.

| Config       | Key change(s)               |    ROUGE-1 |    ROUGE-2 |    ROUGE-L |   Runtime | Notes                                                   |
| ------------ | --------------------------- | ---------: | ---------: | ---------: | --------: | ------------------------------------------------------- |
| Baseline V1  | Zero-shot GPT-2 (Prompt V1) |     0.0377 |     0.0015 |     0.0310 |         — | gen_len=54.3                                            |
| Baseline V2  | Zero-shot GPT-2 (Prompt V2) |     0.0632 | **0.0054** |     0.0437 |         — | gen_len=79.4                                            |
| A            | V1, r=8, LR=5e-4            |     0.0578 |     0.0000 |     0.0328 | 12.44 min | Underperforms; likely unstable/underfit                 |
| B            | V1, r=8, LR=2e-4            |     0.0809 | **0.0026** |     0.0462 | 12.44 min | Best ROUGE-2 among V1 runs                              |
| C            | V1, r=16, LR=5e-4           |     0.0750 |     0.0013 |     0.0430 | 12.45 min | Lower across ROUGE metrics                              |
| D            | V1, r=16, LR=2e-4           |     0.0786 |     0.0018 |     0.0464 | 12.45 min | Best ROUGE-L within Prompt V1                           |
| E            | V2, r=8, LR=5e-4            |     0.0829 |     0.0017 |     0.0438 | 12.45 min | Strong ROUGE-1, weaker ROUGE-L                          |
| F            | V2, r=8, LR=2e-4            |     0.0772 |     0.0000 |     0.0462 | 12.46 min | Good ROUGE-L but ROUGE-2 = 0                            |
| G            | V2, r=16, LR=5e-4           |     0.0832 | **0.0026** |     0.0450 | 12.46 min | Best ROUGE-2 among fine-tuned V2 (but not best ROUGE-L) |
| **H (Best)** | **V2, r=16, LR=2e-4**       | **0.0864** |     0.0011 | **0.0504** | 12.47 min | Best ROUGE-1 and ROUGE-L overall                        |


## 7) Best config: why it won (metrics + tradeoffs)

* **Best config:** **Config H** (Prompt V2 + LoRA r=16 + LR=2e-4).

* **What improved (numbers):**

  * Versus the best Prompt V1 fine-tune on ROUGE-L (**Config D, 0.0464**), Config H reached **ROUGE-L = 0.0504** (**+0.0040**).
  * Versus the best Prompt V1 fine-tune on ROUGE-1 (**Config B, 0.0809**), Config H reached **ROUGE-1 = 0.0864** (**+0.0055**).
  * Versus the **Prompt-matched baseline** (**Baseline V2**):

    * ROUGE-1: **0.0864 − 0.0632 = +0.0232**
    * ROUGE-L: **0.0504 − 0.0437 = +0.0067**
    * ROUGE-2: **0.0011 − 0.0054 = −0.0043**

* **Why it likely improved:**

  * **Prompt V2** already strengthens zero-shot performance (Baseline V2 > Baseline V1), likely by giving clearer summarization constraints and structure.
  * **Higher adapter capacity (r=16)** gives more task-specific capacity under LoRA without full fine-tuning.
  * **Lower LR (2e-4)** tends to be more stable for LoRA SFT, especially with a small dataset subset and short training budget.

* **Tradeoffs / costs:**

  * The best fine-tuned config (H) **did not** maximize ROUGE-2, and the **zero-shot Baseline V2** actually had the highest ROUGE-2 overall. A plausible explanation is that Baseline V2 also generated **much longer outputs** (gen_len 79.4), which can increase n-gram overlap even if summaries are less concise.
  * With a tiny eval set, results are high-variance; manual spot checks (coverage, factuality, brevity) remain important.

* **Where it still fails (likely failure modes):**

  * Missing a key entity/action (e.g., omitting a time, decision, or follow-up).
  * Hallucinating a detail not present in the chat when the dialogue is long/ambiguous.
  * Being verbose or copying chat fragments instead of summarizing.


## 8) How RapidFire AI helped (2–5 bullets)

* Ran the full **2×2×2 (8-run)** SFT grid in one experiment instead of manually coordinating eight separate fine-tuning runs.
* Centralized comparison of **training curves + eval metrics** in TensorBoard artifacts, making it easy to identify the strongest configuration quickly (Config H).
* Reduced sweep overhead (less manual checkpoint/log bookkeeping) and kept the sweep reproducible via consistent configs and a fixed seed.


## 9) Takeaways (3–6 bullets)

* Prompt wording matters for dialogue summarization even **before** training: Baseline V2 outperformed Baseline V1 substantially (and also produced longer generations).
* LoRA rank is a practical capacity knob under Colab constraints; it cleanly trades off adapter capacity vs overfit risk without full fine-tuning.
* In this sweep, **lower LR (2e-4)** tended to be safer, especially at higher rank; Config A suggests **V1 + r8 + higher LR** was brittle under a short budget.
* ROUGE is a useful automatic proxy, but for dialogue summarization it should be complemented by spot checks for factuality/coverage and for excessive copying.
* Next experiment: (i) increase max input length to reduce truncation, and (ii) make eval decoding more controlled (e.g., greedy/beam) to reduce metric variance across runs.
