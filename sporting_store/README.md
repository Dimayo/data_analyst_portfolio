# A/B-тест + кластеризация

[Notebook](research.ipynb) · магазин спортивных товаров

**Стек:** Python, Pandas, NumPy, SciPy, Scikit-Learn, CatBoost, K-Modes, SQLAlchemy, Seaborn

Email-кампания со скидкой, сегментация клиентов и модель склонности к покупке.

## Результат

- **A/B (ARPU):** 26 573 vs 22 271 → **+19.3%**, p < 1e-6, 95% CI [2 723; 5 882]
- **Восстановление пола:** F1 = 0.95
- **Propensity (клиент × товар):** CV ROC-AUC 0.91, holdout ROC-AUC 0.91
- **4 кластера** с рекомендациями по коммуникации

![Сравнение ARPU](images/ab_arpu.png)

## Что сделано

1. **Витрина** — SQLite + CSV, фильтрация по стране, унификация признаков покупок
2. **Gender** — логистическая регрессия по поведению покупок для пропусков
3. **A/B** — Welch t-test по ARPU, валидация через Monte-Carlo (10k разбиений)
4. **Кластеризация** — K-Prototypes, 4 сегмента (премиум / подростки / скидочная женская / низкая чувствительность к скидкам)
5. **Propensity** — CatBoost на парах (клиент, товар), GroupKFold по `id`, top-N для таргетинга

![ROC-кривая](images/model_roc.png)

## Данные

`shop_database.db` (профили, коэффициенты, покупки), `personal_data.csv.gz`, списки treatment/control для кампании.
