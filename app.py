from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import Predictor


st.set_page_config(
    page_title="КардиоРиск AI",
    page_icon="❤️",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Serif:wght@500;600&display=swap');

        :root {
          --bg-soft: #f6fbff;
          --bg-strong: #e6f1ff;
          --card: rgba(255, 255, 255, 0.82);
          --ink: #0f2640;
          --muted: #44607d;
          --accent: #0d8f74;
          --accent-2: #0066cc;
          --danger: #c0392b;
          --shadow: 0 16px 40px rgba(16, 43, 68, 0.12);
          --radius: 18px;
        }

        html, body, [class*="css"] {
          font-family: 'Manrope', sans-serif;
          color: var(--ink);
        }

        .stApp {
          background:
            radial-gradient(1300px 500px at 12% -5%, #dbf2ff 0%, transparent 60%),
            radial-gradient(900px 400px at 96% 14%, #d4e6ff 0%, transparent 55%),
            linear-gradient(160deg, var(--bg-soft) 0%, var(--bg-strong) 100%);
        }

        .block-container {
          max-width: 1180px;
          padding-top: 1.4rem !important;
          padding-bottom: 2.2rem !important;
        }

        .hero {
          background: linear-gradient(125deg, rgba(255, 255, 255, 0.92), rgba(234, 245, 255, 0.95));
          border: 1px solid rgba(84, 115, 146, 0.22);
          border-radius: var(--radius);
          box-shadow: var(--shadow);
          padding: 1.35rem 1.6rem;
          margin-bottom: 1rem;
        }

        .hero h1 {
          font-family: 'IBM Plex Serif', serif;
          font-size: 2rem;
          margin: 0;
          color: #0f2544;
        }

        .hero .subtitle {
          margin-top: 0.15rem;
          color: var(--accent-2);
          font-weight: 700;
          letter-spacing: 0.03em;
          text-transform: uppercase;
          font-size: 0.83rem;
        }

        .hero .description {
          margin-top: 0.7rem;
          color: var(--muted);
          line-height: 1.45;
          font-size: 0.98rem;
        }

        .card {
          background: var(--card);
          border: 1px solid rgba(93, 118, 143, 0.22);
          border-radius: var(--radius);
          box-shadow: var(--shadow);
          padding: 1rem 1.1rem;
        }

        .result-card {
          border-radius: var(--radius);
          padding: 1.1rem 1.2rem;
          border: 1px solid rgba(93, 118, 143, 0.22);
          box-shadow: var(--shadow);
          margin-bottom: 0.8rem;
          background: rgba(255, 255, 255, 0.88);
        }

        .risk-ok {
          border-left: 8px solid var(--accent);
        }

        .risk-alert {
          border-left: 8px solid var(--danger);
        }

        .result-title {
          margin: 0 0 0.3rem 0;
          color: #132f52;
          font-size: 1.1rem;
          font-weight: 800;
        }

        .result-proba {
          font-size: 1.85rem;
          font-weight: 800;
          margin: 0.2rem 0;
        }

        .caption {
          color: var(--muted);
          font-size: 0.9rem;
          margin-top: 0.2rem;
        }

        [data-testid="stForm"] {
          background: rgba(255, 255, 255, 0.84);
          border-radius: var(--radius);
          border: 1px solid rgba(87, 117, 144, 0.25);
          box-shadow: var(--shadow);
          padding: 1.1rem 1rem 0.4rem;
        }

        div[data-testid="stHorizontalBlock"] > div:has(button[kind="secondary"]) button {
          border-radius: 12px;
          border: 1px solid rgba(18, 72, 133, 0.4);
        }

        div[data-testid="stFormSubmitButton"] button {
          border-radius: 12px;
          min-height: 2.7rem;
          font-weight: 700;
        }

        @media (max-width: 920px) {
          .hero h1 { font-size: 1.65rem; }
          .result-proba { font-size: 1.6rem; }
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
    default_profile = predictor.get_default_profile()
    default_profile = normalize_profile_for_inputs(predictor, default_profile)
    for key, value in default_profile.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="subtitle">AI Beyond Fit Predict</div>
          <h1>КардиоРиск AI</h1>
          <div class="description">
            Веб-сервис для оценки риска сердечно-сосудистого события на базе открытого датасета UCI.
            Это учебный инструмент для домашнего задания по vibe-coding.<br>
            <strong>Привет Ивану Стельмаху, спасибо за курс!</strong>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def build_probability_figure(probability: float, threshold: float) -> go.Figure:
    value = round(probability * 100, 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Вероятность риска", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c0392b" if probability >= threshold else "#0d8f74"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#d7f4ed"},
                    {"range": [threshold * 100, 100], "color": "#fde4e1"},
                ],
                "threshold": {
                    "line": {"color": "#1f3e63", "width": 4},
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
    rows = []
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
    colors = ["#c0392b" if x > 0 else "#0d8f74" for x in df["normalized"]]

    fig = go.Figure(
        go.Bar(
            x=df["normalized"],
            y=df["feature_label"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Норм. отклонение: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#6f8bad")
    fig.update_layout(
        title="Сравнение с медианой обучающей выборки",
        xaxis_title="Отклонение (нормированное)",
        yaxis_title="",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def make_recommendations(input_data: dict[str, Any], probability: float) -> list[str]:
    tips: list[str] = []
    if input_data["chol"] >= 240:
        tips.append("Проконсультируйтесь по контролю холестерина: питание, активность, липидный профиль.")
    if input_data["trestbps"] >= 140:
        tips.append("Есть сигнал по артериальному давлению. Полезно обсудить домашний мониторинг давления с врачом.")
    if input_data["oldpeak"] >= 1.5:
        tips.append("Показатель ST-депрессии выше типичного уровня. Нужна очная оценка кардиолога.")
    if input_data["exang"] == 1:
        tips.append("Стенокардия при нагрузке требует аккуратного плана физической активности и диагностики.")
    if input_data["fbs"] == 1:
        tips.append("Есть фактор повышенного сахара натощак. Проверьте глюкозу/гликированный гемоглобин.")

    if not tips and probability < 0.5:
        tips.append("Риск по модели умеренный. Поддерживайте режим сна, ходьбу и регулярные чекапы.")
    if probability >= 0.5 and len(tips) < 3:
        tips.append("Итоговый риск выше порога модели. Лучше пройти очную проверку факторов у врача.")

    return tips[:4]


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
            st.selectbox(
                "Число крупных сосудов (ca)",
                options=[0, 1, 2, 3, 4],
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


def main() -> None:
    inject_styles()
    render_hero()

    try:
        predictor = load_predictor()
    except FileNotFoundError as exc:
        st.error(
            f"{exc}\n\nСначала запустите обучение и сохранение артефактов (например, через `train.ipynb`)."
        )
        st.stop()
    except Exception as exc:  # pragma: no cover - protective branch
        st.error(f"Не удалось загрузить модель: {exc}")
        st.stop()

    initialize_state(predictor)

    demo_profile = {
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

    top_buttons = st.columns([1, 1, 5])
    with top_buttons[0]:
        if st.button("Демо-профиль", type="secondary", use_container_width=True):
            normalized = normalize_profile_for_inputs(predictor, demo_profile)
            for k, v in normalized.items():
                st.session_state[k] = v
            st.rerun()
    with top_buttons[1]:
        if st.button("Сброс", type="secondary", use_container_width=True):
            reset_values = normalize_profile_for_inputs(predictor, predictor.get_default_profile())
            for k, v in reset_values.items():
                st.session_state[k] = v
            st.rerun()

    submitted, payload = render_form(predictor)

    if not submitted:
        st.info("Заполните параметры и нажмите «Рассчитать».")
        return

    try:
        result = predictor.predict(payload)
    except ValueError as exc:
        st.warning(f"Ошибка валидации: {exc}")
        return
    except Exception as exc:  # pragma: no cover - protective branch
        st.error(f"Ошибка во время расчета: {exc}")
        return

    probability = float(result["risk_probability"])
    threshold = float(result["threshold"])
    label = result["risk_label"]
    risk_percent = probability * 100

    is_high = probability >= threshold
    tone = "risk-alert" if is_high else "risk-ok"
    color = "#c0392b" if is_high else "#0d8f74"

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

    chart_col, explanation_col = st.columns([1.2, 1.3], gap="large")
    with chart_col:
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

    with explanation_col:
        st.markdown("### Почему такой результат")
        factors = result["top_factors"]
        factor_rows = [
            {
                "Фактор": item["feature_label"],
                "Ваше значение": f"{item['value']:.2f}",
                "Медиана train": f"{item['median']:.2f}",
                "Пояснение": item["direction"],
                "Вес признака": f"{item['importance']:.3f}",
            }
            for item in factors
        ]
        st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)

        st.markdown("### Что улучшить")
        for rec in make_recommendations(result["input_echo"], probability):
            st.markdown(f"- {rec}")

        st.markdown(
            """
            <div class="caption">
              Важно: сервис создан в учебных целях и не заменяет медицинскую консультацию.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

