"""Human-readable rendering of family and tokenizer identification results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .config import LABS
from .score import VocabScore
from .store import reported_spend

BAR = "─" * 72


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _lab(name: str) -> str:
    return LABS.get(name, name)


def family_gap_summary(decision: dict) -> str:
    if decision.get("evidence") == "unique tokenizer match among candidates":
        return f" ({decision['family']}, unique exact tokenizer match among candidates)"
    if decision["gap"] is None:
        return ""
    return f" (relative gap {decision['gap']:.2%}, cutoff {decision['cutoff']:.2%})"


def identity_lines(family: dict) -> list[str]:
    if "identity" not in family:
        return []
    identity = family["identity"]
    product, models, tokenizer = (identity[k] for k in ("product_family", "model_candidates", "tokenizer"))
    lines = [f"Model family: {product['family'] or product['verdict']}",
             f"Candidate families: {', '.join(product['candidates']) or 'none'}"]
    if models:
        lines.append("Candidate reference models:")
        for row in models[:8]:
            evidence = [row["family"]]
            if row["distance"] is not None:
                evidence.append(f"distance {row['distance']:.4f}")
            if row["exact_tokenizer"]:
                evidence.append("exact tokenizer")
            lines.append(f"  {row['model']}  ({', '.join(evidence)})")
        if len(models) > 8:
            lines.append(f"  {len(models) - 8} more in JSON")
    lines.append(f"Tokenizer: {tokenizer['verdict']}" +
                 (f" ({', '.join(tokenizer['families'])})" if tokenizer["families"] else ""))
    if claim := family.get("model_claim"):
        detail = (f" (distance {claim['distance']:.4f}, cutoff {claim['cutoff']:.4f})"
                  if claim["distance"] is not None else "")
        lines.append(f"Supplied model claim: {claim['claimed_model']} — {claim['verdict']}{detail}")
    return lines


def render_vocab_table(scores: Sequence[VocabScore], limit: int = 8) -> str:
    if not scores:
        return "  (no vocabulary comparison available)"
    lines = [f"  {'vocabulary':<18}{'lab':<16}{'score':>7}{'exact':>9}{'MAE':>7}"]
    for s in scores[:limit]:
        lines.append(
            f"  {s.key:<18}{_lab(s.lab):<16}{_pct(s.score):>7}"
            f"{f'{s.exact}/{s.n}':>9}{s.mae:>7.2f}"
        )
    return "\n".join(lines)


def render_misses(score: VocabScore, limit: int = 10) -> str:
    if not score.misses:
        return "  exact on every probe"
    items = sorted(score.misses.items(), key=lambda kv: -abs(kv[1]))[:limit]
    return "  worst probe errors (measured − predicted): " + ", ".join(
        f"{name}{value:+d}" for name, value in items
    )


def unavailable_status(records) -> str:
    if records and all(r.status in {404, 410} for r in records):
        return "unavailable"
    return "no_model_response"


def render_unavailable(*, target: str, endpoint: str, model: str,
                       records: Sequence, calls: int = 0) -> str:
    messages: List[str] = []
    counts: dict = {}
    for rec in records:
        text = (rec.text or "").strip()
        if not text:
            continue
        if text not in counts:
            messages.append(text)
        counts[text] = counts.get(text, 0) + 1
    messages.sort(key=lambda m: -counts[m])

    out = [BAR,
           f"target    {target}   ({endpoint} / {model})",
           f"requests  {calls}   reported spend ${reported_spend(records):.4f}",
           BAR, "",
           "VERDICT: LISTING UNAVAILABLE" if unavailable_status(records) == "unavailable" else "VERDICT: NO MODEL RESPONSE",
           f"  no probe succeeded: 0 of {len(records)} requests returned a model response",
           ""]
    if messages:
        out.append("WHAT THE ENDPOINT SAID")
        for msg in messages[:2]:
            out.append(f"  {msg[:300]}")
        out.append("")
    out.append(BAR)
    return "\n".join(out)


def sieve_lines(sieve: dict) -> list[str]:
    l1 = sieve.get("layer1") or {}
    cands = ", ".join(l1.get("candidates") or []) or "none"
    mae = f"{l1['min_mae']:.4f}" if l1.get("min_mae") is not None else "n/a"
    verdict = sieve["verdict"].upper()
    headline = f"VERDICT: {verdict}" + (f" -> {sieve['family']}" if sieve.get("family") else "")
    lines = [
        headline,
        f"  mode     : {sieve['method']}",
        f"  evidence : {sieve['evidence']}",
        "",
        "LAYER 1 — DIFFERENTIAL TOKENIZER SIEVE",
        f"  probes measured : {l1.get('probes', 0)}/76   min MAE : {mae}",
        f"  candidate class : {cands}",
    ]
    nearest = l1.get("nearest") or []
    if nearest:
        lines.append(f"  {'family':<18}{'MAE':>8}  nearest enrolled reference")
        for row in nearest[:5]:
            lines.append(f"  {row['family']:<18}{row['mae']:>8.4f}  {row['neighbor']}")
    if l2 := sieve.get("layer2"):
        lines.append("")
        lines.append("LAYER 2 & 3 — COLLISION DISCRIMINATOR (REASONING + CHAT TEMPLATE)")
        tel = l2.get("telemetry") or {}
        if tel:
            lines.append(
                f"  stage 2 probes  : {tel.get('prompts_ok', l2.get('n_ok', 0))}/16   "
                f"median prompt tokens : {tel.get('median_prompt_tokens', 'n/a')}"
            )
            lines.append(
                f"  reasoning       : active on {tel.get('reasoning_active', 0)}/{tel.get('prompts_ok', 0)} prompts "
                f"(mean {tel.get('mean_reasoning_tokens', 0.0):.1f} tokens)"
            )
            if tel.get("sample_random"):
                lines.append(f"  random anchor   : {tel['sample_random']}")
        by_b = {r["family"]: r for r in (l2.get("behavioral") or [])}
        by_t = {r["family"]: r for r in (l2.get("template") or [])}
        by_c = {r["family"]: r for r in (l2.get("combined") or [])}
        fams = [r["family"] for r in (l2.get("combined") or l2.get("behavioral") or [])]
        if fams:
            lines.append(f"  {'family':<16}{'behav':>8}{'templ':>8}{'comb':>8}  nearest reference (behav / templ)")
            for f in fams[:6]:
                db = f"{by_b[f]['distance']:.4f}" if f in by_b else "n/a"
                dt = f"{by_t[f]['distance']:.4f}" if f in by_t else "n/a"
                dc = f"{by_c[f]['distance']:.4f}" if f in by_c else "n/a"
                nb = by_b[f]["neighbor"] if f in by_b else "-"
                nt = by_t[f]["neighbor"] if f in by_t else "-"
                ref_col = nb if nb == nt else f"{nb} / {nt}"
                lines.append(f"  {f:<16}{db:>8}{dt:>8}{dc:>8}  {ref_col}")
    return lines


def render_sieve(sieve: dict, *, target: str, endpoint: str, calls: int = 0,
                 spend: float = 0.0, store: Optional[str] = None,
                 providers: Optional[dict] = None) -> str:
    out = [
        BAR,
        f"target    {target}   ({endpoint})",
        f"requests  {calls}   reported spend ${spend:.4f}",
        BAR,
        "",
        *sieve_lines(sieve),
        "",
    ]
    if providers:
        routes = ", ".join(f"{name} ({count})" for name, count in sorted(providers.items()))
        out.append(f"Gateway-reported routes: {routes}")
    if store:
        out.append(f"Recording: {store}")
    out.append(BAR)
    return "\n".join(out)


def render(*, target: str, endpoint: str, model: str,
           vocab_scores: Sequence[VocabScore] = (),
           catalog: Optional[Dict[str, Any]] = None,
           calls: int = 0, spend: float = 0.0,
           extra_notes: Sequence[str] = (), family: Optional[dict] = None,
           sieve: Optional[dict] = None,
           providers: Optional[dict] = None) -> str:
    out: List[str] = []
    out.append(BAR)
    out.append(f"target    {target}   ({endpoint} / {model})")
    out.append(f"requests  {calls}   reported spend ${spend:.4f}")
    out.append(BAR)
    out.append("")

    if sieve:
        out.extend(sieve_lines(sieve))
        out.append("")

    if family:
        nearest = ", ".join(family["leading_families"]) or "insufficient observations"
        out.append(f"MODEL FAMILY: {family['decision']['verdict'].upper()}{family_gap_summary(family['decision'])}")
        out.extend(identity_lines(family))
        out.append(f"  nearest enrolled: {nearest} ({family['method']}, {family['attempts']} requests)")
        for item in family["ranked"]:
            if item["family"] in family["leading_families"]:
                out.append(f"  nearest reference for {item['family']}: {item['neighbor']}")
        out.append("")

    if vocab_scores:
        out.append("TOKENIZER LINEAGE")
        out.append(render_vocab_table(vocab_scores))
        out.append(render_misses(vocab_scores[0]))
        out.append("")

    if catalog:
        out.append("CATALOGUE METADATA")
        out.append(f"  tokenizer family declared : {catalog.get('tokenizer_declared')}")
        if catalog.get("hugging_face_id"):
            out.append(f"  hugging_face_id           : {catalog['hugging_face_id']}")
        if reasoning := (catalog.get("reasoning") or {}):
            out.append(f"  reasoning                 : {reasoning}")
        if catalog.get("supported_parameters"):
            out.append(f"  supported parameters      : {', '.join(catalog['supported_parameters'])}")
        out.append("")

    if providers:
        routes = ", ".join(f"{name} ({count})" for name, count in sorted(providers.items()))
        out.extend([f"Gateway-reported routes: {routes}", ""])
    if extra_notes:
        out.append("NOTES")
        for note in extra_notes:
            out.append(f"  · {note}")
        out.append("")

    out.append(BAR)
    return "\n".join(out)
