"""Render a private matched-validation study, including negative findings."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from src.common import digest, read_json, write_json
from src.evaluation.evaluate import evaluate_predictions, evaluate_search, evaluate_recommendation, search_request_metrics
from src.evaluation.v2_study import identity, jsonl


def request_key(row):
    return str(row["user_raw"]), str(row["request_id"])


def paired_user_difference(left, right, seed=42, repeats=2000):
    """User-cluster bootstrap for the predeclared user-macro NDCG endpoint."""
    a = {request_key(r): search_request_metrics(r) for r in left}
    b = {request_key(r): search_request_metrics(r) for r in right}
    if set(a) != set(b):
        raise ValueError("Paired comparison requires identical search requests")
    by_user = defaultdict(list)
    for key in a:
        if a[key]["exclusion"] != b[key]["exclusion"]:
            raise ValueError("Paired eligibility differs")
        if a[key]["exclusion"] is None:
            by_user[key[0]].append(b[key]["ndcg"]-a[key]["ndcg"])
    values = np.asarray([np.mean(by_user[user]) for user in sorted(by_user)])
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(values, (min(100, repeats-i), len(values)), replace=True).mean(1)
                            for i in range(0, repeats, 100)])
    return {"direction": "TRACE final minus comparator", "mean_delta": float(values.mean()),
            "ci95": np.percentile(draws, [2.5,97.5]).tolist(), "users": len(values),
            "repeats": repeats, "seed": seed,
            "scope": "exploratory validation interval; no seed variance or multiple-comparison correction"}


def summarize(study, history_path):
    study = Path(study)
    branches = read_json(study / "branches/branch-results.json")
    baselines = read_json(study / "baselines/baseline-results.json")
    methods = {**{name: {"metrics": entry["metrics"], "prediction": study / f"baselines/{name}/validation.jsonl"}
                  for name,entry in baselines.items()},
               **{f"TRACE {name}": {"metrics": branches["metrics"][name], "prediction": study / f"branches/{name}.jsonl"}
                  for name in ("base", "always", "final")}}
    # Post-hoc diagnostic motivated by a narrow learned-gate distribution:
    # compare selection to an ordinary fixed blend using saved logits only.
    # It does not retrain, tune a blend weight, or replace the selected model.
    fixed_path = study / "branches/fixed-half.jsonl"
    with fixed_path.open("w", encoding="utf-8") as handle:
        for base, always in zip(jsonl(study / "branches/base.jsonl"), jsonl(study / "branches/always.jsonl"), strict=True):
            if identity(base) != identity(always):
                raise ValueError("Branch identities differ for fixed blend")
            key = "scores" if base["task"] == "search" else "score"
            values = .5 * (np.asarray(base[key]) + np.asarray(always[key]))
            base[key] = values.tolist()
            handle.write(json.dumps(base, allow_nan=False) + "\n")
    methods["TRACE fixed gate 0.5"] = {"metrics": evaluate_predictions(jsonl(fixed_path)), "prediction": fixed_path}
    diagnostics = list(jsonl(study / "branches/requests.jsonl"))
    search_diags = {request_key(r): r for r in diagnostics if r["task"] == "search"}
    exposure_known = {}
    exposure_labels = {}
    for row in diagnostics:
        if row["task"] == "recommendation":
            for raw, known, label in zip(row["item_raw"], row["item_known"], row["labels"]):
                key = (str(row["user_raw"]), str(row["request_id"]), str(raw))
                exposure_known[key] = known
                exposure_labels[key] = label
    del diagnostics
    searches = {}
    slice_results = {}
    for name, entry in methods.items():
        search, rec_labels, rec_scores = [], defaultdict(list), defaultdict(list)
        seen_exposures = set()
        for row in jsonl(entry["prediction"]):
            if row["task"] == "search":
                search.append(row)
            else:
                key = (str(row["user_raw"]), str(row["request_id"]), str(row["item_raw"]))
                if key in seen_exposures or key not in exposure_labels or row["label"] != exposure_labels[key]:
                    raise ValueError(f"Recommendation identity/label alignment mismatch: {name}")
                seen_exposures.add(key)
                known = exposure_known[key]
                group = "known_item" if known else "oov_item"
                rec_labels[group].append(row["label"])
                rec_scores[group].append(row["score"])
        if seen_exposures != set(exposure_labels):
            raise ValueError(f"Incomplete recommendation candidate coverage: {name}")
        searches[name] = search
        sliced = defaultdict(list)
        for row in search:
            d = search_diags[request_key(row)]
            for group, selected in {
                "cross_history_present": d["cross_history"], "cross_history_absent": not d["cross_history"],
                "history_0": d["history_blocks"] == 0, "history_1_10": 0 < d["history_blocks"] <= 10,
                "history_11_50": d["history_blocks"] > 10,
                "candidates_1_10": d["candidates"] <= 10, "candidates_11_50": 10 < d["candidates"] <= 50,
                "candidates_over_50": d["candidates"] > 50,
                "query_empty": not d["query_present"],
                "query_known_below_half": d["query_present"] and d["query_known_fraction"] < .5,
                "query_known_at_least_half": d["query_present"] and d["query_known_fraction"] >= .5,
                "raw_overlap_absent": d["max_raw_query_caption_overlap"] == 0,
                "raw_overlap_present": d["max_raw_query_caption_overlap"] > 0,
                "items_all_known": d["oov_fraction"] == 0,
                "items_mixed": 0 < d["oov_fraction"] < 1,
                "items_all_oov": d["oov_fraction"] == 1,
            }.items():
                if selected:
                    sliced[group].append(row)
            for known in (True, False):
                indices = [i for i,v in enumerate(d["item_known"]) if v == known]
                if indices:
                    subset = {**row, **{key: [row[key][i] for i in indices] for key in ("item_raw", "labels", "scores")}}
                    sliced["within_known_candidates" if known else "within_oov_candidates"].append(subset)
        slice_results[name] = {"search": {group: evaluate_search(rows) for group,rows in sliced.items()},
            "recommendation": {group: evaluate_recommendation(rec_labels[group], rec_scores[group]) for group in rec_labels}}
    reference = searches["TRACE final"]
    ref_by_key = {request_key(r): r for r in reference}
    for name, rows in searches.items():
        if {request_key(r) for r in rows} != set(ref_by_key):
            raise ValueError(f"Search identities differ for {name}")
        for row in rows:
            ref = ref_by_key[request_key(row)]
            if row["item_raw"] != ref["item_raw"] or row["labels"] != ref["labels"]:
                raise ValueError("Complete candidate/label alignment mismatch")
    paired = {name: paired_user_difference(rows, reference) for name, rows in searches.items() if name != "TRACE final"}
    utility = {}
    for task in ("search", "recommendation"):
        groups = defaultdict(list)
        for d in jsonl(study / "branches/requests.jsonl"):
            if d["task"] != task or not d["cross_history"]:
                continue
            delta = d["loss"]["base"]-d["loss"]["always"]
            if task == "search" and not (len(set(d["item_raw"])) >= 2 and 0 < sum(d["labels"]) < len(d["labels"])):
                continue
            group = "helpful" if delta >= .005 else "harmful" if delta <= -.005 else "ambiguous"
            groups[group].append(d["gate"])
        utility[task] = {group: {"requests": len(v), "mean_gate": float(np.mean(v))} for group,v in groups.items()}
    table = [{"method": name, "search_user_ndcg": entry["metrics"]["search"]["overall"]["user_macro"]["ndcg@10"],
              "search_request_ndcg": entry["metrics"]["search"]["overall"]["request_macro"]["ndcg@10"],
              "rec_ap": entry["metrics"]["recommendation"]["average_precision"],
              "rec_log_loss": entry["metrics"]["recommendation"]["log_loss"]}
             for name,entry in methods.items()]
    result = {"scope": "10k cohort, validation only, seed 42; no new neural training", "table": table,
        "primary_endpoint": "overall user-macro search NDCG@10", "parity": branches["parity"],
        "gate": branches["gate_cross_history_only"], "utility_validation_diagnostic": utility,
        "paired_user_bootstrap": paired, "slices": slice_results,
        "prediction_hashes": {name: digest(entry["prediction"]) for name,entry in methods.items()},
        "complete_search_and_recommendation_identity_label_alignment": True,
        "test_accessed": False, "branch_interventions": "frozen, not retrained; fixed-half blend is a post-hoc diagnostic",
        "baseline_convergence": {name: v.get("converged") for name,v in baselines.items()}}
    write_json(study / "comparison.json", result)
    make_figures(study, result, read_json(history_path))
    lines = ["# TRACE V2 — matched validation evaluation", "", "Date: 5 November 2026, Asia/Jakarta.", "",
        "10,000-user prepared cohort; seed 42; test outcomes remain untouched. The selected TRACE checkpoint is update 4,000, chosen by the original equal-task validation BCE criterion.", "",
        "## Matched comparison", "", "| Method | Search user NDCG@10 | Search request NDCG@10 | Rec AP | Rec log loss |",
        "|---|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f'| {row["method"]} | {row["search_user_ndcg"]:.4f} | {row["search_request_ndcg"]:.4f} | {row["rec_ap"]:.4f} | {row["rec_log_loss"]:.4f} |')
    lines += ["", "Primary endpoint: search **user-macro** NDCG@10 (higher is better). Recommendation AP is supporting; lower log loss is better. All methods use identical validation requests, full candidate lists and labels. Tabular baselines use the complete training split once, fixed C=1/max_iter=300 and task/group weights; TRACE used 10k stochastic neural updates. This is a quality comparator, not an equal-update or equal-compute neural architecture study.", "",
        "Logistic metadata features include the original safe features and raw query–caption coverage/Jaccard matching TRACE's available lexical information. The no-caption sensitivity removes caption/metadata/raw-overlap features while retaining query/history information. Frequency/overlap is a fixed rule with training-only task intercepts. No hyperparameter sweep or test-based selection was performed.", "",
        "## Frozen branch comparison", "", "Base/always/final share the selected trained weights. Disabling or forcing the gate at inference diagnoses this checkpoint; these interventions are not retrained controls and cannot isolate the effect of utility supervision. A post-hoc fixed-gate-0.5 diagnostic derives an untuned 50/50 blend from saved base/always logits because the learned gate has a narrow distribution. It does not replace the selected model.", "",
        "| Comparator | TRACE final minus comparator: user NDCG | Exploratory 95% user-bootstrap interval |", "|---|---:|---:|"]
    for name, entry in paired.items():
        lines.append(f'| {name} | {entry["mean_delta"]:+.5f} | [{entry["ci95"][0]:+.5f}, {entry["ci95"][1]:+.5f}] |')
    lines += ["", "Intervals resample 900 eligible users with replacement (2,000 draws, seed 42), conditional on this validation cohort and trained seed. They do not establish robustness across training seeds; multiple-comparison correction is not applied.", "",
        "## Verification and coverage", "", f'CPU export parity passed over {branches["parity"]["candidates"]:,} candidate scores. Maximum absolute GPU/CPU logit difference: {branches["parity"]["max_abs_logit_error"]:.3g}; fixed tolerance atol=rtol=0.0002. Saved candidates/labels/identities were checked before accepting parity.', "",
        "Coverage tables in comparison.json include history/candidate/query bins, seen/OOV items, same-service/transition cohorts and recommendation known/OOV slices. Within-known/OOV search metrics rerank only that candidate subset and have different eligibility denominators; they are diagnostics, not replacements for the primary full-candidate score. Transition search has only 194 eligible requests, so its findings remain exploratory.", "",
        "## Remaining gates", "", "The strongest matched comparator, feature coverage and fixed-branch results determine the next focused correction. A utility-supervision contribution still needs a retrained ordinary-gate/no-utility control, other required controls and repeated seeds. Freeze the final model/protocol before opening final test outcomes. CPU inference here is offline batch evaluation, not a measured production latency SLA.", "",
        "![Matched validation comparison](comparison.png)", "", "![Validation curves](validation-curves.png)", "",
        "Detailed metrics: [comparison.json](comparison.json). Baseline and branch ledgers retain prediction checksums, convergence information, timing and feature provenance. No Git commit, push or external publication was made."]
    (study / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def make_figures(study, result, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    names = [r["method"] for r in result["table"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, layout="constrained")
    for ax, key, title in zip(axes, ("search_user_ndcg", "rec_ap", "rec_log_loss"),
                             ("Search user NDCG@10 ↑", "Recommendation AP ↑", "Recommendation log loss ↓")):
        values = [r[key] for r in result["table"]]
        bars = ax.barh(names, values, color=["#15706a" if n == "TRACE final" else "#9faebd" for n in names])
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_xlim(0, max(values)*1.25)
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("TRACE V2 · matched 10k-user validation · one seed")
    fig.savefig(study / "comparison.png")
    plt.close(fig)
    values = [r for r in history if "validation" in r]
    x = [r["step"] for r in values]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    ys = ([r["validation"]["search"]["overall"]["user_macro"]["ndcg@10"] for r in values],
          [r["validation"]["recommendation"]["average_precision"] for r in values],
          [r["validation"]["joint_log_loss"] for r in values])
    for ax, y, title in zip(axes, ys, ("Search user NDCG@10 ↑", "Recommendation AP ↑", "Equal-task validation BCE ↓")):
        ax.plot(x, y, marker="o", color="#15706a")
        ax.axvline(4000, linestyle="--", color="#a34826", label="Selected checkpoint")
        ax.set(title=title, xlabel="Optimizer update")
        ax.grid(alpha=.25)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Checkpoint selected by BCE; test outcomes remain closed")
    fig.savefig(study / "validation-curves.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.study, args.history)
    print(json.dumps({"table": result["table"], "parity": result["parity"]}, indent=2))
