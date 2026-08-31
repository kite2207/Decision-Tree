"""Verify saved Decision tree improvement artifacts against the held-out dataset."""

import unittest

import joblib
import numpy as np
import pandas as pd

from src.improvements.tree_experiments import BASE_DIR, GRIDS, evaluate, load_data


class ImprovementArtifactsTest(unittest.TestCase):
    def test_saved_models_and_metrics(self):
        data = load_data()
        results = pd.read_csv(BASE_DIR / "dataset/improvements_results.csv").set_index("model")
        predictions = pd.read_csv(BASE_DIR / "dataset/improvements_predictions.csv")
        self.assertEqual(set(results.index), {"baseline", *GRIDS})
        np.testing.assert_array_equal(predictions["actual"], data[3])
        for name in results.index:
            with self.subTest(model=name):
                model = joblib.load(BASE_DIR / f"model/improvements_{name}.joblib")
                metrics, predicted = evaluate(model, *data)
                np.testing.assert_array_equal(predictions[name], predicted)
                for metric, value in metrics.items():
                    self.assertAlmostEqual(results.loc[name, metric], value)
                self.assertEqual(sum(metrics[k] for k in ("tn", "fp", "fn", "tp")), len(data[3]))
                self.assertEqual(metrics["nodes"], 2 * metrics["leaves"] - 1)

    def test_selection_uses_cv_accuracy(self):
        results = pd.read_csv(BASE_DIR / "dataset/improvements_results.csv").set_index("model")
        for name, grid in GRIDS.items():
            with self.subTest(experiment=name):
                cv = pd.read_csv(BASE_DIR / f"dataset/improvements_cv_{name}.csv")
                self.assertEqual(len(cv), len(next(iter(grid.values()))))
                self.assertTrue(np.isfinite(cv["mean_test_accuracy"]).all())
                self.assertAlmostEqual(results.loc[name, "cv_accuracy"], cv["mean_test_accuracy"].max())
                for fold in range(5):
                    self.assertIn(f"split{fold}_test_accuracy", cv)


if __name__ == "__main__":
    unittest.main()
