"""Safe-feature, train-only fitting and balanced-task baseline contracts."""
from copy import deepcopy
import pickle
import unittest

import numpy as np

from src.training.tabular import (
    CheapBaseline, SafeFeatureExtractor, TabularBaseline,
    prediction_records, task_balanced_weights,
)


def record(*, task="search", name="r", items=("10", "20"), labels=(1, 0),
           captions=((2, 3), (4,)), query=(2, 3), history=None, split="train"):
    return {"user_raw": "u", "request_id": name, "task": task, "split": split,
            "item_raw": list(items), "item_ids": [2 + index for index in range(len(items))],
            "labels": list(labels), "caption_ids": [list(values) for values in captions],
            "metadata_present": [True] * len(items), "query_ids": list(query),
            "history": history or [], "request_gap": 0, "timestamp_ms": 100,
            "cohort": "ambiguous"}


class SafeFeatureTests(unittest.TestCase):
    def test_exposure_frequency_and_features_do_not_consult_labels(self):
        first = record()
        repeated = record(name="r2", items=("10",), labels=(0,), captions=((2,),))
        changed_labels = deepcopy([first, repeated])
        for row in changed_labels:
            row["labels"] = [1 - label for label in row["labels"]]
        left = SafeFeatureExtractor().fit([first, repeated])
        right = SafeFeatureExtractor().fit(changed_labels)
        self.assertEqual(left.frequency, {"10": 2, "20": 1})
        np.testing.assert_array_equal(left.transform([first]), right.transform([first]))

    def test_validation_transform_does_not_refit_statistics(self):
        fit = SafeFeatureExtractor().fit([record()])
        original = fit.frequency.copy()
        validation = record(split="validation", items=("999",), labels=(1,), captions=((),))
        self.assertEqual(fit.transform([validation])[0, 0], 0)
        self.assertEqual(fit.frequency, original)
        with self.assertRaisesRegex(ValueError, "training records only"):
            fit.fit([validation])

    def test_oov_tokens_and_items_do_not_manufacture_history_overlap(self):
        fit = SafeFeatureExtractor().fit([record()])
        unknown = record(items=("999",), labels=(0,), captions=((1,),), query=(0, 1),
                         history=[{"search_query_ids": [[1]], "rec_item_ids": [1], "service_mask": 3}])
        unknown["item_ids"] = [1]
        feature = dict(zip(fit.feature_names, fit.request_features(unknown)[0]))
        self.assertEqual(feature["query_caption_jaccard"], 0)
        self.assertEqual(feature["query_prior_query_jaccard"], 0)
        self.assertEqual(feature["candidate_in_prior_rec"], 0)
        self.assertTrue(np.isfinite(list(feature.values())).all())

    def test_id_only_removes_caption_and_metadata_dependence(self):
        fit = SafeFeatureExtractor(id_only=True).fit([record()])
        first = record()
        changed = deepcopy(first)
        changed["caption_ids"] = [[90, 91], []]
        changed["metadata_present"] = [False, False]
        np.testing.assert_array_equal(fit.transform([first]), fit.transform([changed]))
        self.assertFalse(any("caption" in name or "metadata" in name for name in fit.feature_names))
        changed["query_ids"] = []
        self.assertFalse(np.array_equal(fit.transform([first]), fit.transform([changed])))

    def test_permuting_unordered_block_contents_preserves_features(self):
        first = record(history=[{"search_query_ids": [[2, 3], [4]], "rec_item_ids": [2, 3], "service_mask": 3}])
        changed = deepcopy(first)
        changed["history"][0]["search_query_ids"] = [[4], [3, 2]]
        changed["history"][0]["rec_item_ids"] = [3, 2]
        fit = SafeFeatureExtractor().fit([first])
        np.testing.assert_array_equal(fit.transform([first]), fit.transform([changed]))

    def test_task_means_balance_groups_and_exposures(self):
        rows = [record(name="s1"),
                record(name="s2", items=("1", "2", "3"), labels=(0, 1, 0), captions=((), (), ())),
                record(task="rec", name="rec", items=("1", "2", "3"), labels=(0, 1, 0), captions=((), (), ()))]
        weights = task_balanced_weights(rows)
        np.testing.assert_allclose(weights[:2].sum(), 2)
        np.testing.assert_allclose(weights[2:5].sum(), 2)
        np.testing.assert_allclose(weights[5:].sum(), 4)
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(task_balanced_weights(rows[:2]).mean(), 1)


class BaselineTests(unittest.TestCase):
    def test_cheap_components_and_pickle_parity(self):
        model = CheapBaseline().fit([record()])
        parts = model.component_scores(record())
        self.assertEqual(parts["token_overlap_scores"], [1.0, 0.0])
        self.assertGreater(parts["scores"][0], parts["scores"][1])
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.predict_request(record()), model.predict_request(record()))
        id_only = CheapBaseline(id_only=True).fit([record()])
        self.assertEqual(id_only.predict_request(record())[0], id_only.predict_request(record())[1])

    def test_logistic_learns_matching_without_id_or_labels_as_features(self):
        rows = [record(name=str(index), items=(str(2 * index), str(2 * index + 1))) for index in range(20)]
        model = TabularBaseline().fit(rows)
        scores = model.predict_request(record(split="validation", items=("999", "1000")))
        self.assertGreater(scores[0], scores[1])
        self.assertLess(np.logaddexp(0, -scores[0]) + np.logaddexp(0, scores[1]), 2 * np.log(2))
        changed = record(split="validation", items=("999", "1000"), labels=(0, 1))
        self.assertEqual(model.predict_request(changed), scores)
        scaler_mean = model.scaler.mean_.copy()
        model.predict_request(changed)
        np.testing.assert_array_equal(model.scaler.mean_, scaler_mean)

    def test_tree_classifier_uses_fixed_budget_and_returns_logits(self):
        rows = [record(name=str(index)) for index in range(30)]
        model = TabularBaseline(model="trees").fit(rows)
        self.assertEqual(model.classifier.n_iter_, 60)
        self.assertFalse(model.classifier.early_stopping)
        scores = model.predict_request(record(split="validation"))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0], scores[1])

    def test_prediction_export_preserves_candidates_and_rec_exposures(self):
        model = CheapBaseline().fit([record()])
        rows = [record(split="validation"), record(task="rec", name="rec", split="validation")]
        output = prediction_records(model, rows)
        self.assertEqual(output[0]["item_raw"], rows[0]["item_raw"])
        self.assertEqual(output[0]["labels"], rows[0]["labels"])
        self.assertEqual(len(output), 3)
        self.assertEqual([row["label"] for row in output[1:]], [1, 0])
        self.assertEqual([row["item_raw"] for row in output[1:]], ["10", "20"])
        self.assertTrue(all(row["task"] == "recommendation" for row in output[1:]))

    def test_constant_training_labels_fail_instead_of_fabricating_model(self):
        with self.assertRaisesRegex(ValueError, "both labels"):
            TabularBaseline().fit([record(labels=(0, 0))])
        with self.assertRaisesRegex(RuntimeError, "fit training"):
            CheapBaseline().predict_request(record())


if __name__ == "__main__":
    unittest.main()
