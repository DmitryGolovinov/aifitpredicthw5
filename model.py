from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class Predictor:
    """Inference wrapper for CardioRisk AI model."""

    def __init__(
        self,
        model_path: str | Path = "artifacts/model.joblib",
        meta_path: str | Path = "artifacts/meta.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {self.model_path}. Сначала обучите модель."
            )
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"Файл метаданных не найден: {self.meta_path}. Сначала обучите модель."
            )

        self.model = joblib.load(self.model_path)
        with self.meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.feature_order: list[str] = self.meta["feature_order"]
        self.validation_rules: dict[str, dict[str, Any]] = self.meta["validation_rules"]
        self.feature_labels: dict[str, str] = self.meta.get("feature_labels", {})

    def get_default_profile(self) -> dict[str, float | int]:
        """Returns defaults based on medians and allowed values in metadata."""
        medians = self.meta.get("medians", {})
        defaults: dict[str, float | int] = {}
        for feature in self.feature_order:
            rule = self.validation_rules.get(feature, {})
            allowed = rule.get("allowed")
            if allowed:
                median_val = float(medians.get(feature, allowed[0]))
                defaults[feature] = min(allowed, key=lambda x: abs(float(x) - median_val))
            else:
                value = float(medians.get(feature, rule.get("min", 0.0)))
                if rule.get("type") == "int":
                    defaults[feature] = int(round(value))
                else:
                    defaults[feature] = float(value)
        return defaults

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculates risk probability and explanation for a single observation."""
        validated = self._validate_input(data)
        frame = pd.DataFrame([validated], columns=self.feature_order)

        if hasattr(self.model, "predict_proba"):
            risk_probability = float(self.model.predict_proba(frame)[0][1])
        else:
            risk_probability = float(self.model.predict(frame)[0])

        threshold = float(self.meta.get("threshold", 0.5))
        risk_label = (
            "Высокий риск сердечно-сосудистого события"
            if risk_probability >= threshold
            else "Низкий/умеренный риск сердечно-сосудистого события"
        )

        return {
            "risk_probability": risk_probability,
            "risk_label": risk_label,
            "threshold": threshold,
            "top_factors": self._build_top_factors(validated, top_n=3),
            "input_echo": validated,
        }

    def _validate_input(self, data: dict[str, Any]) -> dict[str, float | int]:
        if not isinstance(data, dict):
            raise ValueError("Ожидался словарь с входными признаками.")

        missing = [feature for feature in self.feature_order if feature not in data]
        if missing:
            missing_ru = ", ".join(self.feature_labels.get(f, f) for f in missing)
            raise ValueError(f"Не заполнены обязательные поля: {missing_ru}.")

        cleaned: dict[str, float | int] = {}
        for feature in self.feature_order:
            rule = self.validation_rules.get(feature, {})
            cleaned[feature] = self._coerce_value(feature, data[feature], rule)

        return cleaned

    def _coerce_value(
        self,
        feature: str,
        raw_value: Any,
        rule: dict[str, Any],
    ) -> float | int:
        if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
            feature_name = self.feature_labels.get(feature, feature)
            raise ValueError(f"Поле '{feature_name}' не должно быть пустым.")

        feature_name = self.feature_labels.get(feature, feature)
        expected_type = rule.get("type", "float")

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Поле '{feature_name}' должно быть числом.") from exc

        if expected_type == "int":
            if not numeric_value.is_integer():
                raise ValueError(f"Поле '{feature_name}' должно быть целым числом.")
            converted: float | int = int(numeric_value)
        else:
            converted = float(numeric_value)

        if "allowed" in rule:
            allowed_values = list(rule["allowed"])
            if converted not in allowed_values:
                allowed_str = ", ".join(str(v) for v in allowed_values)
                raise ValueError(
                    f"Поле '{feature_name}' имеет недопустимое значение. "
                    f"Разрешено: {allowed_str}."
                )

        min_val = rule.get("min")
        max_val = rule.get("max")
        if min_val is not None and converted < min_val:
            raise ValueError(f"Поле '{feature_name}' должно быть не меньше {min_val}.")
        if max_val is not None and converted > max_val:
            raise ValueError(f"Поле '{feature_name}' должно быть не больше {max_val}.")

        return converted

    def _build_top_factors(
        self,
        row: dict[str, float | int],
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        importances = self.meta.get("feature_importance", {})
        medians = self.meta.get("medians", {})
        ranges = self.meta.get("ranges", {})

        scored_factors: list[dict[str, Any]] = []
        for feature in self.feature_order:
            importance = float(importances.get(feature, 0.0))
            value = float(row[feature])
            median = float(medians.get(feature, 0.0))
            range_info = ranges.get(feature, {})
            span = float(range_info.get("max", median)) - float(range_info.get("min", median))
            span = max(span, 1e-9)

            deviation = value - median
            normalized_deviation = deviation / span
            impact_score = abs(normalized_deviation) * importance

            if abs(deviation) < 1e-9:
                direction = "на уровне медианы"
            elif deviation > 0:
                direction = "выше медианы"
            else:
                direction = "ниже медианы"

            scored_factors.append(
                {
                    "feature": feature,
                    "feature_label": self.feature_labels.get(feature, feature),
                    "value": value,
                    "median": median,
                    "direction": direction,
                    "importance": importance,
                    "impact_score": impact_score,
                }
            )

        scored_factors.sort(key=lambda x: x["impact_score"], reverse=True)
        return scored_factors[:top_n]

