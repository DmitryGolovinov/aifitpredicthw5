# КардиоРиск AI

Учебный ML-веб-сервис для домашнего задания по курсу **AI Beyond Fit Predict**.

- Датасет: [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Задача: бинарная классификация риска сердечно-сосудистого события (`num > 0`)
- Интерфейс: Streamlit (русский язык, интерактивные графики, объяснение факторов)

В интерфейсе проекта есть сообщение:
**«Привет Ивану Стельмаху, спасибо за курс!»**

## Структура проекта

- `train.ipynb` — обучение модели и сохранение артефактов
- `model.py` — класс `Predictor` с методом `predict(data)`
- `app.py` — Streamlit-приложение
- `artifacts/model.joblib` — сохранённая модель
- `artifacts/meta.json` — метаданные (валидация, медианы, важности, порог)
- `requirements.txt` — зафиксированные зависимости
- `Dockerfile` — контейнеризация для локального запуска и Timeweb Cloud

## Локальный запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Приложение откроется на `http://localhost:8501`.

## Пересборка модели (опционально)

Если нужно переобучить и пересоздать артефакты, откройте и выполните `train.ipynb` целиком.

На выходе должны обновиться:

- `artifacts/model.joblib`
- `artifacts/meta.json`

## Smoke-check для `Predictor`

```bash
python -c "from model import Predictor; p=Predictor(); print(p.predict({'age':57,'sex':1,'cp':4,'trestbps':140,'chol':241,'fbs':0,'restecg':1,'thalach':123,'exang':1,'oldpeak':2.0,'slope':2,'ca':1,'thal':7}))"
```

## Автотесты валидации

```bash
python -m unittest tests/test_predictor.py -v
```

## Docker

### Сборка и запуск локально

```bash
docker build -t cardio-risk-ai .
docker run --rm -p 8501:8501 cardio-risk-ai
```

## Деплой в Timeweb Cloud (App Platform + Dockerfile)

1. Создайте новый GitHub-репозиторий и запушьте этот проект в ветку `main`.
2. Войдите в [Timeweb Cloud](https://timeweb.cloud/) и откройте раздел **Apps**.
3. Нажмите **Создать приложение**.
4. Выберите источник **GitHub** и подключите аккаунт (если ещё не подключён).
5. Выберите ваш репозиторий и ветку `main`.
6. В способе сборки выберите **Dockerfile** (путь: `Dockerfile` в корне проекта).
7. В переменных окружения можно добавить `PORT=8501` (или оставить по умолчанию, т.к. в `Dockerfile` есть fallback `${PORT:-8501}`).
8. Запустите деплой и дождитесь статуса **Running**.
9. Откройте выданный домен приложения и проверьте форму + кнопку **Рассчитать**.

## Что можно показать проверяющему

- работающий веб-сервис с русским UI;
- код `Predictor` в `model.py`;
- Dockerfile;
- репозиторий на GitHub;
- скриншот запущенного сервиса (локально или по Timeweb-ссылке).

---

Важно: сервис учебный и не заменяет медицинскую консультацию.
