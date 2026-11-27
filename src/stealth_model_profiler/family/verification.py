"""Verify one supplied model claim against its enrolled observations."""
import hashlib
from itertools import groupby

from .detector import distances


VERIFY_REPETITIONS = 15


def equal_error_cutoff(trials):
    """Select an inclusive distance threshold, keeping tied scores together."""
    ordered = sorted((score, genuine) for score, genuine in trials if score is not None)
    positives = sum(genuine for _, genuine in ordered)
    negatives = len(ordered) - positives
    if not positives or not negatives:
        return None
    accepted_true = accepted_false = 0
    # Distances are nonnegative. Include the operating point that rejects all.
    best = (1., 0., -1.)
    for score, group in groupby(ordered, key=lambda trial: trial[0]):
        for _, genuine in group:
            accepted_true += genuine
            accepted_false += not genuine
        far = accepted_false / negatives
        frr = 1 - accepted_true / positives
        best = min(best, (abs(far - frr), far, score))
    return best[2]


def calibrate_claims(even, odd):
    # Reserve two thirds of model identities for evaluation, including all
    # impostor trials involving those identities.
    ordered = sorted(even.keys() & odd.keys(), key=lambda m: hashlib.sha256(m.encode()).digest())
    calibration = ordered[:len(ordered) // 3]
    trials = ((distances(odd[p], even[r])["combined"], p == r) for p in calibration for r in calibration)
    return {"method": "combined", "cutoff": equal_error_cutoff(trials),
            "calibration_models": calibration, "references": even}


def claim_reference(bank, model):
    calibration = bank.get("claim_verification", {})
    if model not in bank["models"]:
        return None
    return calibration.get("references", {}).get(model, bank["models"][model])


def has_claim_reference(bank, model):
    reference = claim_reference(bank, model) or {}
    return (bank.get("claim_verification", {}).get("cutoff") is not None
            and sum(sum(row["answers"].values()) >= 10 for row in reference.values()) >= 20)


def verify_claim(cells, bank, model):
    reference = claim_reference(bank, model)
    cutoff = bank.get("claim_verification", {}).get("cutoff")
    score = None
    if reference is None:
        verdict = "not enrolled"
    elif cutoff is None:
        verdict = "insufficient calibration"
    else:
        score = distances(cells, reference)["combined"]
        if score is None:
            verdict = "insufficient evidence"
        else:
            verdict = "accepted" if score <= cutoff else "rejected"
    return dict(claimed_model=model, verdict=verdict, distance=score, cutoff=cutoff,
                method="answer distributions and input-token counts")
