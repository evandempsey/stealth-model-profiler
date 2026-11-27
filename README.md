# stealth-model-profiler

Identify the model family behind an OpenRouter stealth listing while the model is still anonymous, using short-answer behavior and token-count fingerprints.

## Install

Requires Python 3.11+.

```bash
python -m pip install .
stealth-model-profiler --help
```

Set `OPENROUTER_API_KEY` for live measurements (or pass `--env-file` / `MPROF_ENV_FILE`; by default the CLI checks `./.env` and `~/.config/model-profiler/.env`).

## Quick start

```bash
stealth-model-profiler family-refs
stealth-model-profiler stealth --no-check
stealth-model-profiler sieve --model stealth/EXAMPLE --json
```

`family-refs` downloads the checksum-pinned PAMELA study archives (~53 MB) and builds a single local reference index covering 56 named product families and 57 tokenizer fingerprints.

`sieve` runs the **3-layer sieve**:
1. **Layer 1 (Tokenizer Sieve):** runs the 76-probe tokenizer panel (`max_tokens=1`, works on mandatory-reasoning models) and narrows to families within `MAE ≤ 0.15`. Unique matches return immediately; unseen tokenizers abstain.
2. **Layer 2 (Reasoning & Behavior):** only when Layer 1 hits a shared-tokenizer collision, runs 16 deterministic (`temp=0`) prompts to measure reasoning-token budget curves and answer character 4-grams.
3. **Layer 3 (Chat-Template Offset):** compares raw prompt-token framing offsets inside the collision group. Default `--mode strict` calls only when Layer 2 and Layer 3 agree (`--mode behavioral` ignores prompt-token offsets; `--mode combined` fuses both).

See [METHOD.md](METHOD.md) for the full method overview and blind evaluation results across OpenRouter and Vercel AI Gateway.

Use `--json -` for stdout or `--json PATH` to save a JSON report.

## Commands

| Command | Purpose |
| --- | --- |
| `family-refs` | Build or update the local family reference index. |
| `stealth [--no-check]` | List OpenRouter stealth slots and flag newly appeared slugs. |
| `sieve --model SLUG` | Run the 3-layer tokenizer + reasoning sieve (or `--from-run PATH` offline). |
| `profile --model SLUG` | Run the 40-prompt family battery + tokenizer probes and save a recording. |
| `replay PATH` | Re-score a saved JSONL recording offline (`--refresh-catalog` fetches live metadata). |
| `family --model SLUG` | Run the 40-prompt family battery directly (or `--from-run PATH` offline). |
| `tokens --model SLUG` | Compare token-count deltas over 76 strings against pinned public vocabularies. |
| `catalog SLUG` | Show OpenRouter's declared metadata and endpoints for a slug. |

Live commands accept `--max-calls`, `--max-spend`, and `--concurrency`.

### Verify a specific model claim

```bash
stealth-model-profiler family --model stealth/EXAMPLE --verify-model openai/gpt-4.1 --json
stealth-model-profiler replay runs/RECORDING.jsonl --verify-model openai/gpt-4.1 --json -
```

`--verify-model` runs a separate claim check (15 draws across 40 prompts for eligible live claims) without changing the family verdict.

## Decision rules & limits

- **Default (`predictive`):** compares short answers + prompt-token counts across 40 prompts (needs at least 20 comparable prompts). Accepts a unique named family when its distance is **>5% below** the runner-up.
- **Counts-only (`counts` / mandatory-reasoning routes):** compares prompt-token counts alone and requires a **>20%** gap.
- **Tokenizer tiebreak:** on an abstention, a complete tokenizer fingerprint can resolve the tie when every plausible family has a measured reference and exactly one matches.
- **Caveats:** an unenrolled family can resemble an enrolled one; token counts alone can reflect a shared vocabulary or gateway accounting.

## Data credit

Built on the PAMELA study corpus: [Study data (CC BY 4.0)](https://doi.org/10.5281/zenodo.21278557) and [Study protocol & labels (MIT)](https://doi.org/10.5281/zenodo.21278793), plus supplemental fingerprints bundled with the package.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests
```

MIT License.
