from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import Predictor


st.set_page_config(
    page_title="КардиоРиск AI",
    page_icon="🫀",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Literata:opsz,wght@7..72,600&display=swap');

        :root {
          --bg-main: #f6fbff;
          --bg-soft: #eaf3ff;
          --ink: #10253d;
          --muted: #46627f;
          --card: rgba(255, 255, 255, 0.88);
          --line: rgba(84, 113, 140, 0.28);
          --accent: #0b8f75;
          --danger: #c0392b;
          --brand: #0e5ac7;
          --radius: 18px;
          --shadow: 0 18px 45px rgba(17, 41, 68, 0.12);
        }

        html, body, [class*="css"] {
          font-family: 'Sora', sans-serif;
          color: var(--ink);
        }

        .stApp {
          background:
            radial-gradient(1200px 520px at 4% -8%, #d9ecff 0%, transparent 58%),
            radial-gradient(900px 470px at 96% 8%, #d7e8ff 0%, transparent 58%),
            linear-gradient(140deg, var(--bg-main) 0%, var(--bg-soft) 100%);
        }

        .block-container {
          max-width: 1220px;
          padding-top: 1.35rem !important;
          padding-bottom: 2.2rem !important;
        }

        .hero {
          border-radius: var(--radius);
          border: 1px solid var(--line);
          box-shadow: var(--shadow);
          background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(231,242,255,0.96));
          padding: 1.4rem 1.45rem;
          margin-bottom: 0.95rem;
          animation: rise 0.45s ease-out;
        }

        .hero h1 {
          margin: 0.18rem 0 0;
          font-family: 'Literata', serif;
          font-size: 2.08rem;
          letter-spacing: 0.01em;
          color: #10223e;
        }

        .hero .subtitle {
          color: var(--brand);
          letter-spacing: 0.06em;
          text-transform: uppercase;
          font-size: 0.82rem;
          font-weight: 700;
        }

        .hero .description {
          margin-top: 0.62rem;
          font-size: 0.96rem;
          color: var(--muted);
          line-height: 1.52;
        }

        .metric-strip {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.62rem;
          margin-top: 0.8rem;
        }

        .metric-pill {
          border: 1px solid var(--line);
          border-radius: 14px;
          background: rgba(255,255,255,0.78);
          padding: 0.58rem 0.72rem;
        }

        .metric-pill .label {
          color: var(--muted);
          font-size: 0.78rem;
          margin-bottom: 0.18rem;
        }

        .metric-pill .value {
          font-weight: 800;
          color: #132b4a;
          font-size: 1rem;
        }

        .guide-card {
          border-radius: 16px;
          border: 1px solid var(--line);
          background: rgba(255,255,255,0.83);
          padding: 0.7rem 0.9rem;
          margin-bottom: 0.88rem;
          box-shadow: 0 10px 28px rgba(23, 48, 72, 0.08);
        }

        .result-card {
          border-radius: var(--radius);
          border: 1px solid var(--line);
          box-shadow: var(--shadow);
          padding: 1rem 1.1rem;
          margin-bottom: 0.75rem;
          background: rgba(255, 255, 255, 0.92);
        }

        .risk-alert {
          border-left: 8px solid var(--danger);
        }

        .risk-ok {
          border-left: 8px solid var(--accent);
        }

        .result-title {
          margin: 0;
          font-size: 1rem;
          color: #17375a;
          font-weight: 700;
        }

        .result-proba {
          font-size: 1.98rem;
          margin: 0.2rem 0 0.08rem;
          font-weight: 800;
        }

        .caption {
          color: var(--muted);
          font-size: 0.89rem;
        }

        [data-testid="stForm"] {
          border-radius: 18px;
          border: 1px solid var(--line);
          background: rgba(255, 255, 255, 0.86);
          box-shadow: var(--shadow);
          padding: 1rem 1rem 0.45rem;
        }

        [data-testid="stTabs"] button[role="tab"] {
          font-weight: 700;
        }

        div[data-testid="stFormSubmitButton"] button,
        .stButton > button {
          border-radius: 12px;
          min-height: 2.65rem;
          font-weight: 700;
          border: 1px solid rgba(29, 70, 113, 0.3);
        }

        @keyframes rise {
          from { transform: translateY(10px); opacity: 0.8; }
          to { transform: translateY(0); opacity: 1; }
        }

        @media (max-width: 980px) {
          .hero h1 { font-size: 1.7rem; }
          .metric-strip { grid-template-columns: 1fr; }
          .result-proba { font-size: 1.62rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_predictor() -> Predictor:
    return Predictor()


def normalize_profile_for_inputs(predictor: Predictor, values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    rules = predictor.validation_rules
    for feature in predictor.feature_order:
        rule = rules.get(feature, {})
        allowed = rule.get("allowed")
        raw_value = values.get(feature)
        if allowed:
            if raw_value is None:
                normalized[feature] = allowed[0]
            else:
                normalized[feature] = min(allowed, key=lambda x: abs(float(x) - float(raw_value)))
            continue
        if raw_value is None:
            raw_value = rule.get("min", 0.0)
        if rule.get("type") == "int":
            normalized[feature] = int(round(float(raw_value)))
        else:
            normalized[feature] = float(raw_value)
    return normalized


def initialize_state(predictor: Predictor) -> None:
    default_profile = normalize_profile_for_inputs(predictor, predictor.get_default_profile())
    for key, value in default_profile.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_demo_profile() -> dict[str, Any]:
    return {
        "age": 63,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 278,
        "fbs": 1,
        "restecg": 0,
        "thalach": 130,
        "exang": 1,
        "oldpeak": 2.4,
        "slope": 2,
        "ca": 2,
        "thal": 7,
    }


def build_random_profile(predictor: Predictor) -> dict[str, Any]:
    rng = np.random.default_rng(seed=42 + datetime.now().second)
    profile: dict[str, Any] = {}
    for feature in predictor.feature_order:
        rule = predictor.validation_rules.get(feature, {})
        allowed = rule.get("allowed")
        if allowed:
            profile[feature] = int(rng.choice(allowed))
            continue
        min_v = float(rule.get("min", 0))
        max_v = float(rule.get("max", min_v + 1))
        if rule.get("type") == "int":
            profile[feature] = int(rng.integers(int(min_v), int(max_v) + 1))
        else:
            profile[feature] = round(float(rng.uniform(min_v, max_v)), 1)
    return profile


def render_hero(predictor: Predictor) -> None:
    metrics = predictor.meta.get("metrics", {})
    accuracy = float(metrics.get("accuracy", 0.0))
    roc_auc = float(metrics.get("roc_auc", 0.0))
    train_rows = int(predictor.meta.get("train_shape", [0, 0])[0])
    st.markdown(
        f"""
        <section class="hero">
          <div class="subtitle">AI Beyond Fit Predict</div>
          <h1>КардиоРиск AI</h1>
          <div class="description">
            Приложение для оценки риска сердечно-сосудистого события на открытом датасете UCI.
            <strong>Привет Ивану Стельмаху, спасибо за курс!</strong>
          </div>
          <div class="metric-strip">
            <div class="metric-pill"><div class="label">ROC-AUC holdout</div><div class="value">{roc_auc:.3f}</div></div>
            <div class="metric-pill"><div class="label">Accuracy holdout</div><div class="value">{accuracy:.3f}</div></div>
            <div class="metric-pill"><div class="label">Train samples</div><div class="value">{train_rows}</div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_guide() -> None:
    st.markdown(
        """
        <div class="guide-card">
          <strong>Гайд: что интересного можно показать на защите</strong>
          <div class="caption">Ниже вкладки с экспериментами: what-if анализ, сравнение сценариев и объяснение вклада факторов.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Инструкция для демо (4 шага)", expanded=True):
        st.markdown(
            """
            1. Нажми `Демо-профиль` и затем `Рассчитать`.
            2. Во вкладке `Сценарии` сравни текущий риск с профилем, приближенным к медианам train.
            3. Во вкладке `Лаборатория факторов` выбери любой признак и посмотри интерактивную кривую риска.
            4. Скачай итог через кнопку `Скачать отчёт JSON` и приложи скрин в домашку.
            """
        )


def build_probability_figure(probability: float, threshold: float) -> go.Figure:
    value = round(probability * 100, 1)
    color = "#c0392b" if probability >= threshold else "#0b8f75"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 35}},
            title={"text": "Вероятность риска", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#d7f6ee"},
                    {"range": [threshold * 100, 100], "color": "#fde4e1"},
                ],
                "threshold": {
                    "line": {"color": "#1f3d63", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=310, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_comparison_figure(predictor: Predictor, input_data: dict[str, Any]) -> go.Figure:
    medians = predictor.meta.get("medians", {})
    ranges = predictor.meta.get("ranges", {})
    rows: list[dict[str, Any]] = []
    for feature in predictor.feature_order:
        val = float(input_data[feature])
        median = float(medians.get(feature, 0.0))
        range_info = ranges.get(feature, {})
        span = float(range_info.get("max", median)) - float(range_info.get("min", median))
        span = max(span, 1e-9)
        normalized = (val - median) / span
        rows.append(
            {
                "feature_label": predictor.feature_labels.get(feature, feature),
                "normalized": normalized,
            }
        )

    df = pd.DataFrame(rows)
    df["abs_norm"] = df["normalized"].abs()
    df = df.sort_values("abs_norm", ascending=False).head(8).sort_values("normalized")
    colors = ["#c0392b" if x > 0 else "#0b8f75" for x in df["normalized"]]
    fig = go.Figure(
        go.Bar(
            x=df["normalized"],
            y=df["feature_label"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Норм. отклонение: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#6b87ab")
    fig.update_layout(
        title="Сравнение с медианой обучающей выборки",
        xaxis_title="Отклонение (норм.)",
        yaxis_title="",
        height=330,
        margin=dict(l=10, r=10, t=54, b=10),
    )
    return fig


def build_importance_figure(predictor: Predictor) -> go.Figure:
    raw = predictor.meta.get("feature_importance", {})
    rows = [
        {
            "feature_label": predictor.feature_labels.get(feature, feature),
            "importance": float(score),
        }
        for feature, score in raw.items()
    ]
    df = pd.DataFrame(rows).sort_values("importance", ascending=False).head(10)
    fig = px.bar(
        df.sort_values("importance"),
        x="importance",
        y="feature_label",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        labels={"feature_label": "Признак", "importance": "Важность"},
        title="Глобальная важность признаков",
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=52, b=10), coloraxis_showscale=False)
    return fig


def build_sensitivity_data(
    predictor: Predictor,
    base_profile: dict[str, Any],
    feature: str,
) -> pd.DataFrame:
    rule = predictor.validation_rules.get(feature, {})
    label = predictor.feature_labels.get(feature, feature)
    allowed = rule.get("allowed")
    values: list[float | int]
    if allowed:
        values = list(allowed)
    else:
        ranges = predictor.meta.get("ranges", {})
        min_rule = float(rule.get("min", 0))
        max_rule = float(rule.get("max", min_rule + 1))
        min_train = float(ranges.get(feature, {}).get("min", min_rule))
        max_train = float(ranges.get(feature, {}).get("max", max_rule))
        low = max(min_rule, min_train)
        high = min(max_rule, max_train)
        if high <= low:
            high = low + 1.0
        if rule.get("type") == "int":
            values = sorted(set(int(round(v)) for v in np.linspace(low, high, num=10)))
        else:
            values = [round(float(v), 2) for v in np.linspace(low, high, num=24)]

    rows: list[dict[str, Any]] = []
    for val in values:
        payload = dict(base_profile)
        payload[feature] = val
        proba = predictor.predict_probability(payload)
        rows.append(
            {
                "feature": feature,
                "feature_label": label,
                "value": val,
                "risk_probability": proba,
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity_figure(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["value"],
            y=df["risk_probability"] * 100,
            mode="lines+markers",
            line={"color": "#0e5ac7", "width": 3},
            marker={"size": 8, "color": "#0e5ac7"},
            hovertemplate="Значение: %{x}<br>Риск: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=threshold * 100, line_dash="dash", line_color="#c0392b")
    fig.update_layout(
        title="Лаборатория чувствительности (what-if)",
        xaxis_title="Значение признака",
        yaxis_title="Вероятность риска, %",
        height=360,
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return fig


def build_target_profile(base_profile: dict[str, Any], predictor: Predictor) -> dict[str, Any]:
    medians = predictor.meta.get("medians", {})
    improved = dict(base_profile)
    for feature in predictor.feature_order:
        rule = predictor.validation_rules.get(feature, {})
        current = float(base_profile[feature])
        median = float(medians.get(feature, current))
        target = current + (median - current) * 0.7
        allowed = rule.get("allowed")
        if allowed:
            improved[feature] = min(allowed, key=lambda x: abs(float(x) - target))
            continue

        min_v = float(rule.get("min", target))
        max_v = float(rule.get("max", target))
        target = min(max(target, min_v), max_v)
        if rule.get("type") == "int":
            improved[feature] = int(round(target))
        else:
            improved[feature] = round(target, 1)
    return improved


def make_recommendations(input_data: dict[str, Any], probability: float) -> list[str]:
    tips: list[str] = []
    if input_data["chol"] >= 240:
        tips.append("Проконтролировать липидный профиль и питание для снижения холестерина.")
    if input_data["trestbps"] >= 140:
        tips.append("Вести дневник давления и обсудить результаты с врачом.")
    if input_data["oldpeak"] >= 1.5:
        tips.append("Показатель ST-депрессии требует углубленной консультации кардиолога.")
    if input_data["exang"] == 1:
        tips.append("Стенокардия при нагрузке: снизить интенсивные нагрузки до очной диагностики.")
    if input_data["fbs"] == 1:
        tips.append("Проверить глюкозу/гликированный гемоглобин и метаболические факторы.")
    if probability >= 0.5 and len(tips) < 3:
        tips.append("Итоговый риск выше порога модели, лучше пройти очный чек-ап.")
    if not tips:
        tips.append("Поддерживайте умеренную физическую активность, сон и регулярный профилактический контроль.")
    return tips[:5]


def render_form(predictor: Predictor) -> tuple[bool, dict[str, Any]]:
    rules = predictor.validation_rules
    with st.form("cardio_form"):
        st.markdown("### Ввод параметров пациента")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.number_input(
                "Возраст",
                min_value=int(rules["age"]["min"]),
                max_value=int(rules["age"]["max"]),
                step=1,
                key="age",
            )
            st.selectbox(
                "Пол",
                options=[0, 1],
                format_func=lambda x: "Женщина" if x == 0 else "Мужчина",
                key="sex",
            )
            st.selectbox(
                "Тип боли в груди",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Типичная стенокардия",
                    2: "Атипичная стенокардия",
                    3: "Несердечная боль",
                    4: "Бессимптомно",
                }[x],
                key="cp",
            )
            st.number_input(
                "Давление в покое (мм рт. ст.)",
                min_value=int(rules["trestbps"]["min"]),
                max_value=int(rules["trestbps"]["max"]),
                step=1,
                key="trestbps",
            )
            st.number_input(
                "Холестерин (mg/dl)",
                min_value=int(rules["chol"]["min"]),
                max_value=int(rules["chol"]["max"]),
                step=1,
                key="chol",
            )

        with col2:
            st.selectbox(
                "Сахар натощак > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "Нет" if x == 0 else "Да",
                key="fbs",
            )
            st.selectbox(
                "ЭКГ в покое",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "Норма",
                    1: "Нарушения ST-T",
                    2: "Гипертрофия ЛЖ",
                }[x],
                key="restecg",
            )
            st.number_input(
                "Максимальный пульс",
                min_value=int(rules["thalach"]["min"]),
                max_value=int(rules["thalach"]["max"]),
                step=1,
                key="thalach",
            )
            st.selectbox(
                "Стенокардия при нагрузке",
                options=[0, 1],
                format_func=lambda x: "Нет" if x == 0 else "Да",
                key="exang",
            )
            st.slider(
                "Снижение ST (oldpeak)",
                min_value=float(rules["oldpeak"]["min"]),
                max_value=float(rules["oldpeak"]["max"]),
                step=0.1,
                key="oldpeak",
            )

        with col3:
            st.selectbox(
                "Наклон ST-сегмента",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "Восходящий",
                    2: "Плоский",
                    3: "Нисходящий",
                }[x],
                key="slope",
            )
            st.number_input(
                "Число крупных сосудов (ca)",
                min_value=0,
                max_value=4,
                step=1,
                key="ca",
            )
            st.selectbox(
                "Thal",
                options=[3, 6, 7],
                format_func=lambda x: {
                    3: "Норма",
                    6: "Фиксированный дефект",
                    7: "Обратимый дефект",
                }[x],
                key="thal",
            )

        submitted = st.form_submit_button("Рассчитать")

    payload = {feature: st.session_state[feature] for feature in predictor.feature_order}
    return submitted, payload


def render_diagnostics_tab(predictor: Predictor, result: dict[str, Any]) -> None:
    probability = float(result["risk_probability"])
    threshold = float(result["threshold"])
    label = result["risk_label"]
    risk_percent = probability * 100
    is_high = probability >= threshold
    tone = "risk-alert" if is_high else "risk-ok"
    color = "#c0392b" if is_high else "#0b8f75"

    st.markdown(
        f"""
        <div class="result-card {tone}">
          <div class="result-title">Итог модели</div>
          <div style="color:{color};" class="result-proba">{risk_percent:.1f}%</div>
          <div><strong>{label}</strong></div>
          <div class="caption">Порог классификации: {threshold:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1.35], gap="large")
    with left:
        st.plotly_chart(
            build_probability_figure(probability, threshold),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            build_comparison_figure(predictor, result["input_echo"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        st.markdown("### Почему такой результат")
        factor_rows = [
            {
                "Фактор": item["feature_label"],
                "Ваше значение": f"{item['value']:.2f}",
                "Медиана train": f"{item['median']:.2f}",
                "Пояснение": item["direction"],
                "Вес": f"{item['importance']:.3f}",
            }
            for item in result["top_factors"]
        ]
        st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)

        st.markdown("### Что можно улучшить")
        recommendations = make_recommendations(result["input_echo"], probability)
        for tip in recommendations:
            st.markdown(f"- {tip}")

        report_payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": predictor.meta.get("model_name"),
            "dataset": predictor.meta.get("dataset"),
            "result": result,
            "recommendations": recommendations,
        }
        st.download_button(
            "Скачать отчёт JSON",
            data=json.dumps(report_payload, ensure_ascii=False, indent=2),
            file_name=f"cardiorisk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="caption">
              Важно: сервис создан в учебных целях и не заменяет медицинскую консультацию.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_scenarios_tab(predictor: Predictor, base_profile: dict[str, Any]) -> None:
    st.markdown("### Сравнение сценариев")
    improved = build_target_profile(base_profile, predictor)
    base_proba = predictor.predict_probability(base_profile)
    improved_proba = predictor.predict_probability(improved)
    delta = (improved_proba - base_proba) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Текущий риск", f"{base_proba * 100:.1f}%")
    c2.metric("Сценарий «к медианам»", f"{improved_proba * 100:.1f}%")
    c3.metric("Изменение", f"{delta:+.1f} п.п.")

    changed_rows = []
    for feature in predictor.feature_order:
        old = float(base_profile[feature])
        new = float(improved[feature])
        if abs(old - new) < 1e-9:
            continue
        changed_rows.append(
            {
                "Параметр": predictor.feature_labels.get(feature, feature),
                "Было": old,
                "Стало": new,
            }
        )

    if changed_rows:
        st.dataframe(pd.DataFrame(changed_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Профиль уже близок к медианным значениям train.")

    if st.button("Применить сценарий к форме", key="apply_scenario", use_container_width=True):
        normalized = normalize_profile_for_inputs(predictor, improved)
        for key, value in normalized.items():
            st.session_state[key] = value
        st.rerun()


def render_sensitivity_tab(
    predictor: Predictor,
    base_profile: dict[str, Any],
    threshold: float,
) -> None:
    st.markdown("### Лаборатория факторов (what-if)")
    label_to_feature = {predictor.feature_labels.get(f, f): f for f in predictor.feature_order}
    selected_label = st.selectbox(
        "Выберите признак для анализа чувствительности",
        options=list(label_to_feature.keys()),
        index=0,
    )
    selected_feature = label_to_feature[selected_label]
    df = build_sensitivity_data(predictor, base_profile, selected_feature)
    st.plotly_chart(
        build_sensitivity_figure(df, threshold),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    min_row = df.loc[df["risk_probability"].idxmin()]
    max_row = df.loc[df["risk_probability"].idxmax()]
    st.markdown(
        f"""
        - Лучший протестированный уровень `{selected_label}`: **{min_row['value']}** (риск {min_row['risk_probability'] * 100:.1f}%).
        - Худший протестированный уровень `{selected_label}`: **{max_row['value']}** (риск {max_row['risk_probability'] * 100:.1f}%).
        """
    )


def render_insights_tab(predictor: Predictor) -> None:
    st.markdown("### Инсайты модели")
    st.plotly_chart(
        build_importance_figure(predictor),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    metrics = predictor.meta.get("metrics", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC holdout", f"{metrics.get('roc_auc', 0):.3f}")
    c2.metric("Accuracy holdout", f"{metrics.get('accuracy', 0):.3f}")
    c3.metric("Recall (класс риска)", f"{metrics.get('recall_1', 0):.3f}")
    st.caption(f"Источник данных: {predictor.meta.get('dataset', 'UCI Heart Disease')}")


def main() -> None:
    inject_styles()

    try:
        predictor = load_predictor()
    except FileNotFoundError as exc:
        st.error(
            f"{exc}\n\nСначала создайте артефакты через `train.ipynb` (файлы `artifacts/model.joblib` и `artifacts/meta.json`)."
        )
        st.stop()
    except Exception as exc:  # pragma: no cover - protective branch
        st.error(f"Не удалось загрузить модель: {exc}")
        st.stop()

    initialize_state(predictor)
    render_hero(predictor)
    render_guide()

    action_cols = st.columns([1, 1, 1, 5])
    with action_cols[0]:
        if st.button("Демо-профиль", use_container_width=True):
            profile = normalize_profile_for_inputs(predictor, build_demo_profile())
            for key, value in profile.items():
                st.session_state[key] = value
            st.rerun()
    with action_cols[1]:
        if st.button("Случайный профиль", use_container_width=True):
            profile = normalize_profile_for_inputs(predictor, build_random_profile(predictor))
            for key, value in profile.items():
                st.session_state[key] = value
            st.rerun()
    with action_cols[2]:
        if st.button("Сброс", use_container_width=True):
            profile = normalize_profile_for_inputs(predictor, predictor.get_default_profile())
            for key, value in profile.items():
                st.session_state[key] = value
            st.rerun()

    submitted, payload = render_form(predictor)
    payload = normalize_profile_for_inputs(predictor, payload)

    if submitted:
        try:
            result = predictor.predict(payload)
            st.session_state["latest_result"] = result
        except ValueError as exc:
            st.warning(f"Ошибка валидации: {exc}")
        except Exception as exc:  # pragma: no cover - protective branch
            st.error(f"Ошибка во время расчета: {exc}")

    result = st.session_state.get("latest_result")
    if not result:
        st.info("Заполните форму и нажмите «Рассчитать», чтобы открыть все интерактивные фичи.")
        return

    base_profile = result["input_echo"]
    threshold = float(result["threshold"])
    tabs = st.tabs(
        ["Диагностика", "Сценарии", "Лаборатория факторов", "Инсайты модели"],
    )
    with tabs[0]:
        render_diagnostics_tab(predictor, result)
    with tabs[1]:
        render_scenarios_tab(predictor, base_profile)
    with tabs[2]:
        render_sensitivity_tab(predictor, base_profile, threshold)
    with tabs[3]:
        render_insights_tab(predictor)


if __name__ == "__main__":
    main()
