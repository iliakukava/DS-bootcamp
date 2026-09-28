# Кандидатогенерация услуг Авито

Recall@50 на платформе: **0.633**.

## Задача

По запросу вернуть до 50 `item_id` из корпуса объявлений для этапа кандидатогенерации.

## Подход

Гибридный retrieval + эвристики по признакам:

1. **Каналы:** BM25(title), BM25(title+params+prefix desc), char TF-IDF(title), память кликов train, document expansion, dense (`cointegrated/rubert-tiny2`), microcat×location.
2. **Слияние:** взвешенный RRF (k=60).
3. **Пост-обработка:** soft-boost локации/microcat; same-location first; бусты по фильтрам из `search_infm_params_text` (вид/тип услуги, рейтинг).
4. Для редких запросов microcat берётся у ближайшего частого запроса из train (char TF-IDF).

## Данные и признаки

| Признак | Зачем |
| --- | --- |
| `search_query`, `item_title_raw` | основной лексический матч |
| `item_infm_params_text`, prefix `item_description_raw` | слова услуги вне title |
| `search_location_id` / `item_location_id` | ~80% релевантных в той же локации |
| `item_microcat_id` + train | category почти всегда 114; microcat информативнее |
| `search_infm_params_text` | фильтры «вид/тип услуги», рейтинг |
| train click log | память запросов, expansion, microcat-map |

Не использовались: delivery (на тесте всегда 0), category_id, координаты, цена.

## Алгоритмы и библиотеки

- `bm25s`, `pymorphy3`, `scikit-learn` (char TF-IDF)
- `sentence-transformers` + `cointegrated/rubert-tiny2` (локально, без API)
- RRF + rule-based boosts

## Валидация

Group-split по ключу запроса, seed=42, 2400 ключей, доля repeat = 37.5% (как пересечение бенчмарка с train). Корпус val ≈ масштаба бенчмарка. Метрика — Recall@50.

| Версия | Local | Platform |
| --- | ---: | ---: |
| BM25 title | 0.274 | 0.269 |
| + каналы/RRF | 0.434 | 0.379 |
| + dense | 0.445 | 0.391 |
| + geo/filters/rewrite | 0.707 | 0.633 |


## Воспроизведение

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

# dataset/: train.parquet, benchmark_queries.parquet, benchmark_items.parquet
python solve.py
```

Результат: `answer.csv`. Повторный запуск с теми же данными и версиями пакетов даёт тот же файл.
