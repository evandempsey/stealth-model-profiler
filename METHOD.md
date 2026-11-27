# Method: Three-Layer Black-Box Sieve

`stealth-model-profiler` identifies the product family behind an anonymous OpenRouter stealth listing without logprobs, weight access, or disabling model reasoning.

## Why a sieve?

Existing black-box approaches fail in opposite ways on modern API gateways:

- **Answer-distribution methods (e.g., PAMELA)** need up to 1,200 queries (`temp=1`, reasoning disabled), break on mandatory-reasoning models, and suffer high false-accept rates on unseen families.
- **Differential tokenizer fingerprinting (e.g., ModelBehind)** is fast, deterministic, and works on mandatory-reasoning models, but subtracts a baseline prompt to isolate the BPE vocabulary. Whenever multiple product families share a vocabulary (e.g., Gemini/Gemma, Mistral/Nemotron, DeepSeek/Step, Qwen/MiMo/Relace), it must abstain.

## The three layers

### Layer 1 — Near-Exact Differential Tokenizer Sieve
Runs 76 diverse prompt-token probes at `max_tokens=1` (multilingual scripts, code/SQL/diff/LaTeX, digit grouping, whitespace/indentation, ZWJ emoji, Unicode edge cases) against a shared baseline prefix (`"The quick brown fox.\n"`), computing `delta = prompt_tokens(base + probe) - prompt_tokens(base)`.

Rather than requiring brittle 76/76 exact equality (which drops newer model generations whose tokenizer changed on 1–2 probes), Layer 1 requires at least 60 valid probes and selects all enrolled families within `MAE <= 0.15` (and within `0.05` of the best MAE):

- **1 candidate family:** return immediately (zero Stage 2 calls).
- **0 candidate families:** abstain as an unseen tokenizer.
- **2+ candidate families (collision):** pass only that equivalence class to Stage 2, eliminating all vocabulary-incompatible families up front.

### Layer 2 — Reasoning-Budget & Behavioral Profile (inside collisions)
Issues 16 deterministic prompts (`temperature=0`, `max_tokens=512`, reasoning left at default):
- **8 reasoning-effort ladder prompts** (trivial arithmetic, cognitive-reflection trap, letter counting, syllogism, multi-step puzzle, impossible constraint, ambiguous question, short code bug).
- **8 post-training style & preference anchors** (random picks, 3-bullet explanation, Python palindrome, algebra formatting, lowercase conflict, hedging, self-introduction, JSON example).

Compares per-prompt `log(1 + reasoning_tokens)` curves and per-prompt character 4-grams of the final answers. Because it ignores `prompt_tokens`, Layer 2 is completely immune to hidden system prompts or gateway prompt-token padding.

### Layer 3 — Chat-Template Framing Offset (inside collisions)
Compares the **raw, unsubtracted `prompt_tokens`** across the 16 Stage 2 prompts. Differential tokenizer methods discard this offset, yet sibling families sharing a vocabulary almost always wrap user messages with different chat templates—an offset that remains largely invariant across both hosting providers and independent gateways (e.g., Mistral 21.0 vs. Nemotron 34.0–35.0; Gemini 17.5–18.5 vs. Gemma 4 30.5).

### Operating modes
- `--mode strict` *(default)*: calls a family inside a collision only when Layer 2 (behavioral) and Layer 3 (template offset) agree; otherwise abstains.
- `--mode behavioral`: uses Layer 2 only (immune to hidden system prompts).
- `--mode combined`: fuses Layer 2 and Layer 3 distances for higher recall.

## Summary of blind evaluation

All blind test routes are 100% disjoint from the 23 Stage 2 enrollment checkpoints:

| Cohort | Method | Right | Wrong | Abstained | Unmeas. | Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **OpenRouter Validation** (30 routes, incl. 5 unseen labs) | Tokenizer-only | 3 | 1 | 20 | 6 | 75.0% |
| | `sieve --mode strict` | **12** | **2** | 11 | 5 | **85.7%** *(0/5 unseen FP)* |
| **OpenRouter Frozen Blind Test** (25 routes) | Tokenizer-only | 7 | 0 | 15 | 3 | 100.0% *(0 on collisions)* |
| | `sieve --mode strict` | **12** | **2** | 9 | 2 | **85.7%** *(5/1 on collisions)* |
| **Vercel AI Gateway Transfer** (14 routes, OpenRouter refs) | Tokenizer-only | 1 | 0 | 12 | 1 | 100.0% |
| | `sieve --mode strict` | **8** | **0** | 5 | 1 | **100.0%** |
| | `sieve --mode combined` | **11** | **0** | 2 | 1 | **100.0%** |

## Caveats
- **Hidden system prompts** shift Layer 3 (`prompt_tokens`); `--mode strict` catches the Layer 2/Layer 3 disagreement and abstains, while `--mode behavioral` ignores prompt tokens.
- **Missing usage metadata:** endpoints that omit `usage` on `max_tokens=1` requests are reported as unmeasurable rather than guessed.
- **Direct fine-tunes / weight merges:** products directly derived from another family's weights and chat template may match their base family.
