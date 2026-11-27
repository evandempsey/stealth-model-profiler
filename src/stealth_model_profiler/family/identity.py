"""One product-family verdict and the reference models supporting it."""
from .detector import distances
from .tokenizer import compare_signatures
from .novelty import relative_gap


def within_gap(best, distance, cutoff):
    return abs(distance - best) < 1e-12 or (
        cutoff is not None and relative_gap(best, distance) <= cutoff + 1e-12)


def identify(report, cells, enrolled, bank, signature, excluded_model=None):
    decision, ranked = report["decision"], report["ranked"]
    labels = bank["labels"]
    family_cutoff = decision["cutoff"]
    plausible = [r["family"] for r in ranked
                 if within_gap(ranked[0]["distance"], r["distance"], family_cutoff)]
    chosen = None
    if decision["verdict"] == "family match":
        chosen = decision.get("family") or report["leading_families"][0]
    if chosen:
        family_verdict = "match"
    elif decision["verdict"] == "insufficient calibration":
        family_verdict = "insufficient calibration" if len(ranked) >= 2 else "insufficient evidence"
    else:
        family_verdict = "ambiguous" if len(plausible) > 1 else "unknown"
    exact = sorted(m for m, _, match in compare_signatures(signature, bank, excluded_model) if match)
    tokenizer_verdict = "exact" if exact else "no match"
    if signature is None:
        tokenizer_verdict = "unmeasured" if bank.get("tokenizer_signatures") else "no reference signatures"
    exact_families = sorted({labels[m] for m in exact})
    tokenizer = dict(verdict=tokenizer_verdict, models=exact, families=exact_families)
    supported = [chosen] if chosen else plausible
    families = [f for f in (supported or exact_families) if f != "other"]
    family = dict(verdict=family_verdict, family=chosen, candidates=families,
                  evidence=decision.get("evidence", report["method"]) if ranked else "tokenizer" if exact else None,
                  gap=decision["gap"], cutoff=family_cutoff)
    candidates = {}
    for model, reference in enrolled.items():
        if not ranked or labels[model] not in supported:
            continue
        distance = distances(cells, reference)[report["method"]]
        if distance is not None and within_gap(ranked[0]["distance"], distance, family_cutoff):
            candidates[model] = dict(model=model, family=labels[model], distance=distance, exact_tokenizer=model in exact)
    for model in exact:
        if model not in candidates and (not supported or labels[model] in supported):
            candidates[model] = dict(model=model, family=labels[model], distance=None, exact_tokenizer=True)
    models = sorted(candidates.values(), key=lambda r: (r["distance"] is None, r["distance"], r["model"]))
    return dict(product_family=family, model_candidates=models, tokenizer=tokenizer)
