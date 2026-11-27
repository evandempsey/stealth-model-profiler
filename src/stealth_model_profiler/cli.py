"""Command line interface."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

import httpx

from .catalog import fetch_catalog
from .client import Budget, BudgetExceeded, Endpoint
from .config import get_provider, load_credentials
from .family import (
    DEFAULT_GAP_CUTOFFS, collect_family, collect_sieve, family_result,
    recorded_cells, references, sieve_identify,
)
from .family.tokenizer import collect_panel
from .family.verification import VERIFY_REPETITIONS, has_claim_reference, verify_claim
from .features import tokenizer_features
from .probes import BASE, TOKENIZER_PROBES, tokenizer_probes
from .report import (
    family_gap_summary, identity_lines, render, render_sieve, render_unavailable,
    render_vocab_table, unavailable_status,
)
from .runner import run_battery
from .score import group_indistinguishable, rank_vocabularies
from .store import RecordStore, reported_providers, reported_spend, write_json
from .vocab import CACHE_DIR, load_all

FAMILY_INDEX = CACHE_DIR / "pamela" / "reference-extended-coverage.json"


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _store_path(directory: Path, label: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in label)
    return directory / f"{safe}-{_ts()}-{uuid.uuid4().hex[:8]}.jsonl"


def _open_endpoint(args, model: str, store: Path, pin: Optional[str] = None) -> Endpoint:
    provider = get_provider("openrouter")
    if not os.environ.get(provider.key_env):
        raise ValueError(f"{provider.key_env} is required")
    store.parent.mkdir(parents=True, exist_ok=True)
    store.touch(exist_ok=False)
    budget = Budget(max_calls=args.max_calls, max_spend_usd=args.max_spend)
    return Endpoint(provider=provider, model=model, store=RecordStore(store),
                    budget=budget, provider_pin=pin)


def cmd_family_refs(args) -> int:
    data = references.build_index(FAMILY_INDEX)
    print(f"{len(data['models'])} reference fingerprints written to {FAMILY_INDEX}")
    print(data["credit"])
    return 0


def _claim_repetitions(bank, claim):
    if claim and has_claim_reference(bank, claim):
        return VERIFY_REPETITIONS
    return 1


def _family_report(records, bank, model, method, claim):
    family_records = records
    if method == "predictive":
        first = min((int(r.tag.rsplit(":", 1)[1]) for r in records if r.channel == "family"), default=0)
        family_records = [r for r in records if r.channel != "family" or r.tag.endswith(f":{first}")]
    result = family_result(family_records, bank, model, method, exclude_target=False)
    if claim:
        result["model_claim"] = verify_claim(recorded_cells(records, bank["battery"]), bank, claim)
    return result


def cmd_family(args) -> int:
    data = references.load_index(FAMILY_INDEX)
    battery = data["battery"]
    if args.from_run:
        store_path = Path(args.from_run)
        records = RecordStore(store_path).read()
        if len({(r.model, r.endpoint) for r in records}) != 1:
            raise ValueError("supply a nonempty recording of one endpoint and one model")
        model = records[0].model
    else:
        default_reps = _claim_repetitions(data, args.verify_model) if args.method in {"counts", "predictive"} else 15
        reps = args.repetitions if args.repetitions is not None else default_reps
        if args.method in {"combined", "answers"} and reps < 10:
            raise ValueError("answer distributions require at least ten repetitions per prompt")
        if reps < 1 or len(battery) * reps > args.max_calls:
            raise ValueError(f"requested battery needs {len(battery) * reps} calls; check repetitions and --max-calls")
        load_credentials(args.env_file)
        model = args.model
        store_path = _store_path(Path(args.runs), f"family-{model}")
        endpoint = _open_endpoint(args, model, store_path, args.pin)
        try:
            records = collect_family(endpoint, battery, reps, args.method, args.concurrency)
            if "tokenizer_signatures" in data and args.method == "predictive":
                preliminary = _family_report(records, data, model, args.method, None)
                if preliminary["decision"]["verdict"] == "abstain":
                    remaining = endpoint.budget.max_calls - endpoint.budget.calls
                    if remaining >= len(TOKENIZER_PROBES) + 2:
                        try:
                            collect_panel(endpoint, args.concurrency)
                        except BudgetExceeded as exc:
                            print(f"Tokenizer measurement stopped: {exc}", file=sys.stderr)
                        records = endpoint.store.read()
        finally:
            endpoint.close()
    result = _family_report(records, data, model, args.method, args.verify_model)
    result.update(records=str(store_path), reference_file=str(FAMILY_INDEX))
    if args.json:
        if isinstance(args.json, str) and args.json != "-":
            write_json(Path(args.json), result)
        else:
            print(json.dumps(result, ensure_ascii=False))
    else:
        chosen = result["decision"]
        print(f"Method: {result['method']}   Decision: {chosen['verdict']}{family_gap_summary(chosen)}")
        print("\n".join(identity_lines(result)))
        print(f"Closest by distance: {', '.join(result['leading_families']) or 'insufficient observations'}")
        for r in result["ranked"][:5]:
            print(f"  {r['family']:<18} distance {r['distance']:.4f}  {r['neighbor']}")
        print(f"Requests: {len(records)}  Records: {store_path}")
    return 0 if result["leading_families"] else 1


def cmd_sieve(args) -> int:
    bank = references.load_index(FAMILY_INDEX)
    if args.from_run:
        store_path = Path(args.from_run)
        records = RecordStore(store_path).read()
        if len({(r.model, r.endpoint) for r in records}) != 1:
            raise ValueError("supply a nonempty recording of one endpoint and one model")
        model = records[0].model
    else:
        load_credentials(args.env_file)
        model = args.model
        store_path = Path(args.store) if args.store else _store_path(Path(args.runs), f"sieve-{model}")
        endpoint = _open_endpoint(args, model, store_path, args.pin)
        try:
            records = collect_sieve(endpoint, bank, args.concurrency)
        finally:
            endpoint.close()
    result = sieve_identify(records, bank, model, mode=args.mode, exclude_target=False)
    providers = reported_providers(records)
    spend = reported_spend(records)
    result.update(
        records=str(store_path),
        requests=len(records),
        spend_usd=spend,
        gateway_reported_providers=providers,
    )
    if args.json:
        if isinstance(args.json, str) and args.json != "-":
            write_json(Path(args.json), result)
            print(f"report: {args.json}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        ep_name = records[0].endpoint if records else "openrouter"
        print(render_sieve(result, target=model, endpoint=ep_name, calls=len(records),
                           spend=spend, store=str(store_path), providers=providers))
    return 0 if result["verdict"] != "unmeasurable" else 1


def cmd_catalog(args) -> int:
    record = fetch_catalog(args.slug)
    if record is None:
        print(f"{args.slug} is not in the live catalogue", file=sys.stderr)
        return 1
    payload = record.features()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print(f"{record.slug}  —  {record.name}")
    for key, value in payload.items():
        print(f"  {key:<24}{value}")
    if record.endpoints:
        print("  endpoints:")
        for ep in record.endpoints:
            print(f"    · {ep.get('provider_name')}  tag={ep.get('tag')}  "
                  f"quant={ep.get('quantization')}  "
                  f"max_completion={ep.get('max_completion_tokens')}")
    return 0


def cmd_tokens(args) -> int:
    load_credentials(args.env_file)
    store_path = Path(args.store) if args.store else _store_path(Path(args.runs), f"tokens-{args.model}")

    vocab_keys = [k.strip() for k in args.vocabs.split(",")] if args.vocabs else None
    vocabs = load_all(vocab_keys)
    print(f"loaded {len(vocabs)} vocabularies: {', '.join(sorted(vocabs)) or 'none'}")
    if not vocabs:
        print("no vocabularies available; cannot compare", file=sys.stderr)
        return 1

    endpoint = _open_endpoint(args, args.model, store_path, args.pin)
    probes = tokenizer_probes()
    print(f"running {len(probes)} tokenizer probes against {args.model} ...")
    try:
        records = run_battery(endpoint, probes, concurrency=args.concurrency)
    finally:
        endpoint.close()
    feats = tokenizer_features(records)
    if not feats:
        print("tokenizer channel produced no features (all probes failed?)", file=sys.stderr)
        return 1

    scores = rank_vocabularies(feats["deltas"], vocabs.values(), BASE, TOKENIZER_PROBES)
    print()
    print(f"baseline prompt count (text plus chat framing): {feats['base_prompt_tokens']} prompt tokens")
    print(render_vocab_table(scores, limit=12))
    if scores:
        print()
        print(f"best: {scores[0].key} ({scores[0].lab}) — "
              f"{scores[0].exact}/{scores[0].n} exact, MAE {scores[0].mae:.2f}")
        if scores[0].misses:
            worst = sorted(scores[0].misses.items(), key=lambda kv: -abs(kv[1]))[:8]
            print("  worst: " + ", ".join(f"{k}{v:+d}" for k, v in worst))
        for group in group_indistinguishable(scores).values():
            print(f"  identical predictions: {', '.join(group)} (cannot be separated by vocabulary)")
    print(f"\nrequests: {endpoint.budget.calls}   spend: ${endpoint.budget.spend_usd:.4f}")
    print(f"records: {store_path}")
    return 0


def _vocab_scores(args, tok_feats: dict):
    keys = [key.strip() for key in args.vocabs.split(",")] if args.vocabs else None
    deltas = tok_feats.get("deltas", {})
    vocabs = load_all(keys) if deltas else {}
    notes = ["No reference vocabularies could be loaded."] if deltas and not vocabs else []
    return rank_vocabularies(deltas, vocabs.values(), BASE, TOKENIZER_PROBES), notes


def _write_profile_report(destination, recording, payload):
    if destination == "-":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        out = recording.with_suffix(".report.json") if destination in (True, "true") else Path(destination)
        write_json(out, payload)
        print(f"report: {out}")


def _unavailable_profile(args, records, path, model, bank, calls, spend):
    payload = {"target": model, "endpoint": "openrouter", "status": unavailable_status(records),
               "messages": [r.text for r in records if r.text][:3],
               "manifest": {"requests": calls, "records": len(records), "store": str(path), "spend_usd": spend}}
    if args.verify_model:
        payload["family"] = _family_report(records, bank, model, "predictive", args.verify_model)
    if args.json != "-":
        print(render_unavailable(target=model, endpoint="openrouter", model=model, records=records, calls=calls))
        if args.verify_model:
            print("\n".join(identity_lines(payload["family"])))
    if args.json:
        _write_profile_report(args.json, path, payload)
    return 1


def cmd_profile(args) -> int:
    load_credentials(args.env_file)
    store_path = Path(args.store) if args.store else _store_path(Path(args.runs), args.model)

    catalog = None
    if not args.no_catalog:
        record = fetch_catalog(args.model)
        if record:
            catalog = record.features()

    family_bank = references.load_index(FAMILY_INDEX) if (args.verify_model or FAMILY_INDEX.is_file()) else None
    endpoint = _open_endpoint(args, args.model, store_path, args.pin)
    try:
        if family_bank:
            reps = _claim_repetitions(family_bank, args.verify_model)
            if args.verify_model and reps * len(family_bank["battery"]) > args.max_calls:
                raise ValueError("the requested verification battery exceeds --max-calls")
            initial = collect_family(endpoint, family_bank["battery"], reps, "predictive", args.concurrency)
            if initial and all(r.status in {401, 402, 403, 404, 410} for r in initial):
                records = initial
            else:
                if "tokenizer_signatures" in family_bank:
                    collect_panel(endpoint, args.concurrency)
                else:
                    run_battery(endpoint, tokenizer_probes(), concurrency=args.concurrency)
                records = endpoint.store.read()
        else:
            run_battery(endpoint, tokenizer_probes(), concurrency=args.concurrency)
            records = endpoint.store.read()
    finally:
        endpoint.close()

    if not any(r.ok for r in records):
        return _unavailable_profile(args, records, store_path, args.model, family_bank,
                                    endpoint.budget.calls, endpoint.budget.spend_usd)

    vocab_scores, notes = _vocab_scores(args, tokenizer_features(records))
    family = _family_report(records, family_bank, args.model, "predictive", args.verify_model) if family_bank else None
    if family is None:
        notes.append("Install the family reference bank with 'stealth-model-profiler family-refs'.")
    providers = reported_providers(records)
    text = render(family=family, providers=providers, target=args.model, endpoint="openrouter", model=args.model,
                  vocab_scores=vocab_scores, catalog=catalog,
                  calls=endpoint.budget.calls, spend=endpoint.budget.spend_usd,
                  extra_notes=notes)
    if args.json != "-":
        print(text)

    if args.json:
        payload = {
            "target": args.model,
            "endpoint": "openrouter",
            "family": family,
            "gateway_reported_providers": providers,
            "vocab_scores": [
                {"key": s.key, "lab": s.lab, "score": round(s.score, 4),
                 "exact": s.exact, "n": s.n, "mae": round(s.mae, 3)}
                for s in vocab_scores
            ],
            "catalog": catalog,
            "manifest": {"requests": endpoint.budget.calls,
                         "spend_usd": endpoint.budget.spend_usd,
                         "records": len(records), "store": str(store_path)},
        }
        _write_profile_report(args.json, store_path, payload)
    return 0


def cmd_replay(args) -> int:
    records = RecordStore(Path(args.store_path)).read()
    if not records:
        print(f"no records in {args.store_path}", file=sys.stderr)
        return 1
    if len({(r.model, r.endpoint) for r in records}) != 1:
        raise ValueError("replay requires records from one endpoint and model")

    label = records[0].model
    if not any(r.ok for r in records):
        bank = references.load_index(FAMILY_INDEX) if args.verify_model else None
        return _unavailable_profile(args, records, Path(args.store_path), label, bank,
                                    len(records), reported_spend(records))

    catalog = None
    if args.refresh_catalog and not args.no_catalog:
        record = fetch_catalog(label)
        if record:
            catalog = record.features()

    vocab_scores, notes = _vocab_scores(args, tokenizer_features(records))
    family = None
    sieve = None
    has_family = any(r.channel == "family" for r in records)
    has_stage2 = any(r.channel == "stage2" for r in records)
    if args.verify_model or ((has_family or has_stage2) and FAMILY_INDEX.is_file()):
        bank = references.load_index(FAMILY_INDEX)
        if args.verify_model or has_family:
            family = _family_report(records, bank, label, "predictive", args.verify_model)
        if has_stage2:
            sieve = sieve_identify(records, bank, label, mode="strict", exclude_target=False)
    elif has_family or has_stage2:
        notes.append("Install the family reference bank with 'stealth-model-profiler family-refs'.")
    spend = reported_spend(records)
    providers = reported_providers(records)
    if args.json != "-":
        print(render(family=family, sieve=sieve, providers=providers, target=label, endpoint=records[0].endpoint, model=label,
                     vocab_scores=vocab_scores, catalog=catalog,
                     calls=len(records), spend=spend, extra_notes=notes))

    if args.json:
        payload = {
            "target": label,
            "family": family,
            "sieve": sieve,
            "gateway_reported_providers": providers,
            "endpoint": records[0].endpoint,
            "vocab_scores": [
                {"key": s.key, "lab": s.lab, "score": round(s.score, 4),
                 "exact": s.exact, "n": s.n, "mae": round(s.mae, 3)}
                for s in vocab_scores
            ],
            "catalog": catalog,
            "manifest": {"records": len(records), "store": str(args.store_path), "spend_usd": spend},
        }
        _write_profile_report(args.json, Path(args.store_path), payload)
    return 0


def cmd_stealth(args) -> int:
    from .stealth import check_liveness, fetch_slots, load_state, new_slugs, save_state

    try:
        slots = fetch_slots()
    except Exception as exc:
        print(f"could not read the stealth page: {exc}", file=sys.stderr)
        return 2
    if not slots:
        print("no stealth slots found on the page (its shape may have changed)", file=sys.stderr)
        return 2

    state_path = Path(args.state) if args.state else Path(args.runs) / "stealth-slots.json"
    previous = load_state(state_path)

    if not args.no_check:
        load_credentials(args.env_file)
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("no OPENROUTER_API_KEY: skipping liveness check", file=sys.stderr)
        else:
            store = RecordStore(Path(args.runs) / "stealth-watch.jsonl")
            check_liveness(slots, store, max_calls=args.max_calls, max_spend=args.max_spend)

    fresh = new_slugs(previous, slots)

    if args.json:
        print(json.dumps({"slots": [s.to_json() for s in slots], "new": fresh},
                         indent=2, ensure_ascii=False))
    else:
        checked = any(s.live is not None for s in slots)
        if checked:
            live = sum(1 for s in slots if s.live)
            print(f"stealth listings on OpenRouter: {len(slots)} slots, {live} live")
        else:
            print(f"stealth listings on OpenRouter: {len(slots)} slots (liveness not checked)")
        print(f"  {'':4}{'slug':<30}{'name':<20}{'status':<9}reveal")
        for slot in slots:
            status = "LIVE" if slot.live is True else "retired" if slot.live is False else "unknown"
            mark = "NEW" if slot.slug in fresh else ""
            tail = slot.note
            if slot.revealed:
                tail = f"{slot.revealed}"
                if slot.successor:
                    tail += f" -> {slot.successor.split('openrouter.ai/')[-1]}"
            print(f"  {mark:<4}{slot.slug:<30}{slot.name[:19]:<20}{status:<9}{tail}")
        print()
        if not previous:
            print(f"baseline recorded for {len(slots)} slots; a slug appearing later will be flagged NEW")
        elif fresh:
            print(f"{len(fresh)} new since last check: {', '.join(fresh)}")
            for slug in fresh:
                print(f"  stealth-model-profiler profile --model {slug}")
        else:
            print("no change since last check")

    save_state(state_path, slots)
    return 10 if (fresh and args.fail_on_new) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stealth-model-profiler",
        description="Identify candidate model families behind OpenRouter stealth listings.",
    )
    live = argparse.ArgumentParser(add_help=False)
    live.add_argument("--env-file", default=None, help="env file with OPENROUTER_API_KEY")
    live.add_argument("--runs", default="runs", help="directory for raw JSONL records")
    live.add_argument("--max-calls", type=int, default=4000)
    live.add_argument("--max-spend", type=float, default=2.0)
    live.add_argument("--concurrency", type=int, default=4)

    json_opt = argparse.ArgumentParser(add_help=False)
    json_opt.add_argument("--json", nargs="?", const=True, default=False,
                          help="write JSON output (use - for stdout)")

    vocab_opt = argparse.ArgumentParser(add_help=False)
    vocab_opt.add_argument("--vocabs", default=None, help="comma-separated vocabulary keys")

    claim_opt = argparse.ArgumentParser(add_help=False)
    claim_opt.add_argument("--verify-model", metavar="MODEL_ID",
                           help="separately verify a claimed enrolled model ID")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("family-refs", help="install or update the family fingerprint index").set_defaults(func=cmd_family_refs)

    p_cat = sub.add_parser("catalog", parents=[json_opt], help="show OpenRouter catalogue metadata for a slug")
    p_cat.add_argument("slug")
    p_cat.set_defaults(func=cmd_catalog)

    p_family = sub.add_parser("family", parents=[live, json_opt, claim_opt],
                              help="identify the closest model family from short answers and token counts")
    source = p_family.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--from-run", help="replay a family fingerprint recording")
    p_family.add_argument("--pin", help="OpenRouter provider route to pin")
    p_family.add_argument("--method", choices=["predictive", "counts", "answers", "combined"], default="predictive",
                          help=f"scoring method (default cutoffs: {DEFAULT_GAP_CUTOFFS['predictive']:g} predictive, "
                               f"{DEFAULT_GAP_CUTOFFS['counts']:g} counts)")
    p_family.add_argument("--repetitions", type=int, help="draws per prompt")
    p_family.set_defaults(func=cmd_family)

    p_sieve = sub.add_parser("sieve", parents=[live, json_opt],
                             help="identify a model family via the 3-layer tokenizer + reasoning sieve")
    s_src = p_sieve.add_mutually_exclusive_group(required=True)
    s_src.add_argument("--model")
    s_src.add_argument("--from-run", help="replay a saved recording offline")
    p_sieve.add_argument("--mode", choices=["strict", "behavioral", "combined"], default="strict",
                         help="collision-breaking rule (default: strict agreement)")
    p_sieve.add_argument("--store", default=None)
    p_sieve.add_argument("--pin", default=None)
    p_sieve.set_defaults(func=cmd_sieve)

    p_tok = sub.add_parser("tokens", parents=[live, vocab_opt], help="run only the tokenizer probes")
    p_tok.add_argument("--model", required=True)
    p_tok.add_argument("--store", default=None)
    p_tok.add_argument("--pin", default=None)
    p_tok.set_defaults(func=cmd_tokens)

    p_prof = sub.add_parser("profile", parents=[live, json_opt, vocab_opt, claim_opt],
                            help="identify a stealth listing from family and tokenizer fingerprints")
    p_prof.add_argument("--model", required=True, help="model slug, e.g. stealth/ox-alpha")
    p_prof.add_argument("--store", default=None)
    p_prof.add_argument("--pin", default=None)
    p_prof.add_argument("--no-catalog", action="store_true")
    p_prof.set_defaults(func=cmd_profile)

    p_rep = sub.add_parser("replay", parents=[json_opt, vocab_opt, claim_opt],
                           help="re-score a recorded run offline")
    p_rep.add_argument("store_path")
    p_rep.add_argument("--refresh-catalog", action="store_true", help="include current live metadata")
    p_rep.add_argument("--no-catalog", action="store_true")
    p_rep.set_defaults(func=cmd_replay)

    p_st = sub.add_parser("stealth", parents=[live, json_opt],
                          help="list stealth listings and flag newly appeared ones")
    p_st.add_argument("--no-check", action="store_true", help="skip the one-token liveness call")
    p_st.add_argument("--state", default=None, help="snapshot path (default: <runs>/stealth-slots.json)")
    p_st.add_argument("--fail-on-new", action="store_true", help="exit 10 when a new slug appears")
    p_st.set_defaults(func=cmd_stealth)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "max_calls"):
            if args.max_calls < 1 or args.concurrency < 1 or not math.isfinite(args.max_spend) or args.max_spend < 0:
                raise ValueError("call limit and concurrency must be positive, and spend must be finite and nonnegative")
        return args.func(args)
    except BudgetExceeded as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, OSError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
