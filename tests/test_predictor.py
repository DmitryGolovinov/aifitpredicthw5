import unittest

from model import Predictor


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = Predictor()
        cls.valid_payload = {
            "age": 57,
            "sex": 1,
            "cp": 4,
            "trestbps": 140,
            "chol": 241,
            "fbs": 0,
            "restecg": 1,
            "thalach": 123,
            "exang": 1,
            "oldpeak": 2.0,
            "slope": 2,
            "ca": 1,
            "thal": 7,
        }

    def test_predict_smoke(self) -> None:
        result = self.predictor.predict(self.valid_payload)
        self.assertIn("risk_probability", result)
        self.assertIn("risk_label", result)
        self.assertIn("top_factors", result)
        self.assertGreaterEqual(result["risk_probability"], 0.0)
        self.assertLessEqual(result["risk_probability"], 1.0)
        self.assertEqual(len(result["top_factors"]), 3)

    def test_predict_probability_smoke(self) -> None:
        probability = self.predictor.predict_probability(self.valid_payload)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_predict_batch(self) -> None:
        rows = [self.valid_payload, dict(self.valid_payload, age=60)]
        results = self.predictor.predict_batch(rows)
        self.assertEqual(len(results), 2)
        self.assertIn("risk_probability", results[0])

    def test_missing_feature_raises(self) -> None:
        payload = dict(self.valid_payload)
        payload.pop("chol")
        with self.assertRaises(ValueError):
            self.predictor.predict(payload)

    def test_out_of_range_raises(self) -> None:
        payload = dict(self.valid_payload)
        payload["age"] = 999
        with self.assertRaises(ValueError):
            self.predictor.predict(payload)


if __name__ == "__main__":
    unittest.main()
