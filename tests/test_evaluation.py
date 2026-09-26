"""Hand-computed metric fixtures and protocol failure tests."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.evaluation.evaluate import (
    evaluate_predictions, evaluate_recommendation, evaluate_search,
    search_request_metrics,
)


def request(labels, scores, *, items=None, user="u", name="r", cohort="transition"):
    return {"user_raw": user, "request_id": name,
            "item_raw": items if items is not None else list(range(len(labels))),
            "labels": labels, "scores": scores, "cohort": cohort}


class SearchMetricTests(unittest.TestCase):
    def test_hand_computed_correct_and_reverse(self):
        correct = search_request_metrics(request([1, 0], [2, 1]))
        reverse = search_request_metrics(request([1, 0], [1, 2]))
        self.assertEqual(correct["ndcg"], 1.0)
        self.assertAlmostEqual(reverse["ndcg"], 1 / math.log2(3))
        self.assertEqual(reverse["recall"], 1.0)
        self.assertEqual(reverse["mrr"], 0.5)

    def test_raw_numeric_id_ties_are_order_and_label_independent(self):
        # Numeric 2 precedes numeric 10; lexical sorting gives the wrong order.
        first = request([1, 0], [0, 0], items=["10", "2"])
        permuted = request([0, 1], [0, 0], items=["2", "10"])
        self.assertAlmostEqual(search_request_metrics(first)["ndcg"], 1 / math.log2(3))
        self.assertEqual(search_request_metrics(first), search_request_metrics(permuted))

    def test_top_k_cutoff_multiple_clicks_and_mrr(self):
        row = search_request_metrics(request([0, 1, 0, 1], [4, 3, 2, 1]), k=2)
        self.assertAlmostEqual(row["ndcg"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertEqual(row["recall"], 0.5)
        self.assertEqual(row["mrr"], 0.5)
        excluded_click = search_request_metrics(request([0] * 10 + [1], list(range(11, 0, -1))))
        self.assertEqual(excluded_click["ndcg"], 0.0)
        self.assertEqual(excluded_click["recall"], 0.0)
        self.assertEqual(excluded_click["mrr"], 0.0)

    def test_user_macro_differs_from_request_macro(self):
        rows = [request([1, 0], [2, 1], user="a", name="1"),
                request([1, 0], [2, 1], user="a", name="2"),
                request([1, 0], [1, 2], user="b", name="1")]
        result = evaluate_search(rows)["overall"]
        self.assertAlmostEqual(result["user_macro"]["ndcg@10"], (1 + 1 / math.log2(3)) / 2)
        self.assertAlmostEqual(result["request_macro"]["ndcg@10"], (2 + 1 / math.log2(3)) / 3)
        self.assertEqual(result["eligible_users"], 2)
        self.assertEqual(result["eligible_requests"], 3)

    def test_exclusions_are_disjoint_and_remain_in_pointwise_loss(self):
        rows = [request([0], [0], name="one"), request([0, 0], [0, 0], name="zero"),
                request([1, 1], [0, 0], name="all"),
                request([0, 1], [0, 0], user="other", name="eligible")]
        result = evaluate_search(rows)["overall"]
        self.assertEqual(result["exclusions"], {"singleton": 1, "no_positive": 1, "all_positive": 1})
        self.assertEqual(result["eligible_requests"], 1)
        self.assertEqual(result["users_without_eligible_request"], 1)
        self.assertEqual(result["total_candidates"], 7)
        self.assertEqual(result["positive_candidates"], 3)
        self.assertAlmostEqual(result["log_loss"], math.log(2))

    def test_search_loss_is_candidate_weighted(self):
        rows = [request([1], [2], name="one"), request([0, 0], [2, 2], name="two")]
        expected = (math.log1p(math.exp(-2)) + 2 * (2 + math.log1p(math.exp(-2)))) / 3
        self.assertAlmostEqual(evaluate_search(rows)["overall"]["log_loss"], expected)

    def test_cohort_support_is_reported_even_if_empty(self):
        rows = [request([1, 0], [2, 1], name="a", cohort="same_service"),
                request([0, 0], [2, 1], name="b", cohort="ambiguous")]
        result = evaluate_search(rows)
        self.assertEqual(result["same_service"]["eligible_requests"], 1)
        self.assertEqual(result["ambiguous"]["total_requests"], 1)
        self.assertEqual(result["ambiguous"]["eligible_requests"], 0)
        self.assertIsNone(result["transition"]["user_macro"]["ndcg@10"])
        self.assertIsNone(result["ambiguous"]["user_macro"]["ndcg@10"])
        json.dumps(result, allow_nan=False)

    def test_bad_prediction_records_fail(self):
        examples = [request([1, 0], [1]), request([1, 0], [math.nan, 0]),
                    request([1, 2], [1, 0]), request([1, 0], [1, 0], items=["2", 2]),
                    request([1, 0], [1, 0], items=[2.0, 3.0]),
                    request([], []), request([1, 0], [1, 0], cohort="unknown")]
        for row in examples:
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate_search([row])
        with self.assertRaisesRegex(ValueError, "duplicate search request"):
            evaluate_search([request([1, 0], [1, 0])] * 2)
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                evaluate_search([], k=k)


class RecommendationMetricTests(unittest.TestCase):
    def test_noninterpolated_ap_and_roc_hand_fixture(self):
        # Ranked labels 1,0,1,0: AP=(1+2/3)/2, ROC=3/4.
        result = evaluate_recommendation([1, 0, 1, 0], [4, 3, 2, 1])
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertEqual(result["prevalence"], 0.5)

    def test_tied_scores_use_one_threshold(self):
        result = evaluate_recommendation([1, 0, 1, 0], [0, 0, 0, 0])
        self.assertEqual(result["average_precision"], 0.5)
        self.assertEqual(result["roc_auc"], 0.5)
        permuted = evaluate_recommendation([0, 1, 0, 1], [0, 0, 0, 0])
        self.assertEqual(result, permuted)

    def test_extreme_logits_stay_finite_without_sigmoid_ranking(self):
        result = evaluate_recommendation([1, 0, 1, 0], [1000, -1000, -1000, 1000])
        self.assertEqual(result["log_loss"], 500.0)
        huge = evaluate_recommendation([0, 0], [1e308, 1e308])
        self.assertTrue(math.isfinite(huge["log_loss"]))
        self.assertEqual(huge["log_loss"], 1e308)

    def test_empty_and_constant_label_populations_are_explicit(self):
        empty = evaluate_recommendation([], [])
        self.assertIsNone(empty["log_loss"])
        self.assertIsNone(empty["prevalence"])
        self.assertIsNone(empty["average_precision"])
        zeros = evaluate_recommendation([0, 0], [1, 0])
        self.assertIsNone(zeros["average_precision"])
        self.assertIsNone(zeros["roc_auc"])
        ones = evaluate_recommendation([1, 1], [1, 0])
        self.assertEqual(ones["average_precision"], 1.0)
        self.assertIsNone(ones["roc_auc"])

    def test_malformed_inputs_fail(self):
        for labels, scores in (([1], []), ([0.2], [0]), ([1], [math.inf]), (["1"], [0])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                evaluate_recommendation(labels, scores)

    def test_jsonl_cli_mixed_tasks(self):
        records = [request([1, 0], [2, 1]),
                   {"task": "recommendation", "label": 1, "score": 2},
                   {"task": "recommendation", "label": 0, "score": -2}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            output = Path(tmp) / "report.json"
            path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            completed = subprocess.run([sys.executable, "-m", "src.evaluation.evaluate",
                                        "--predictions", str(path), "--output", str(output)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), evaluate_predictions(records))


if __name__ == "__main__":
    unittest.main()
