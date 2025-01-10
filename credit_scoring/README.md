# Модель кредитного риск-менеджмента
[Код проекта в jupyter notebook](https://github.com/Dimayo/credit_scoring/blob/main/scoring_model.ipynb)<br>
Библиотеки python: pandas, numpy, matplotlib, sklearn, joblib 

## Цель проекта
<p>Целью данного проекта является создание одной из моделей оценки кредитного риска – прогнозирования неисполнения клиентом кредита. Дефолт — невыплата процентов по кредиту или облигациям, невозврат кредита в течение определенного времени t. Обычно дефолт считается состоявшимся, если клиент не произвел оплату по кредиту в течение 90 дней.</p> <p>Модель позволяет банку или другой кредитной организации оценить текущий риск по любым выданным кредитам и кредитным продуктам и предотвратить неисполнение клиентом обязательств по кредитным обязательствам с более высокой степенью вероятности. Таким образом, банк меньше рискует понести убытки.</p>

## Описание проекта
<p>Данные содержат информацию о различных характеристиках заемщиков и кредитных продуктах: о клиентах, уже имеющих кредиты, их кредитной истории и финансовых показателях. Каждая запись в наборе данных представляет собой один конкретный кредитный продукт, выданный конкретному заемщику. Метрикой качества используемой модели является roc-auc.</p>

## Что было сделано
Данные большого объема были загружены из нескольких файлов формата parquet. Далее датафрейм был сгруппирован по id с модой в качестве значения, это сделано поскольку у целевого признака только 3 млн. записей:

```
data = data.groupby(['id'], as_index=False).agg(lambda x: pd.Series.mode(x)[0])
```
Далее были созданы новые признаки на основе исходных:
```
data['is_zero_loans'] = np.where((data['is_zero_loans5'] == 1) & (data['is_zero_loans530'] == 1)
                        & (data['is_zero_loans3060'] == 1) & (data['is_zero_loans6090'] == 1)
                        & (data['is_zero_loans90'] == 1), 1, 0)

data['is_zero_overlimit'] = np.where((data['is_zero_over2limit'] == 1)
                            & (data['is_zero_maxover2limit'] == 1), 1, 0)
```
Загружен файл с целевым признаком и объединен с датасетом:
```
data = pd.merge(left=data,right=train_target, how='inner', on='id')
```
Удалены дубликаты в данных где совпадают все строки кроме признака id:
```
data = data.drop_duplicates(subset=data.columns.difference(['id']))
```
Выборка разделена с соблюдением баланса целевого признака:
```
df_train, df_test = train_test_split(data, stratify=data['flag'], test_size=0.3, random_state=42)
```
Т.к целевой признак не сбалансирован был применен метод downsampling:
```
df_min = df_train[df_train['flag'] == 1]
df_maj = df_train[df_train['flag'] == 0]
df_maj_downsample = resample(df_maj, replace=False, n_samples=len(df_min), random_state=42)
df_train= pd.concat([df_maj_downsample, df_min], ignore_index=True).sample(frac=1.)
```
Далее был создан пайплайн для, того чтобы понять оптимальное число признаков и лучшую функцию оценки для модели случайного леса:
```
rf = Pipeline([
    ('selector', GenericUnivariateSelect(mode='k_best')),
    ('rf', RandomForestClassifier())
     ])

params = {'selector__param': np.arange(10,62),
          'selector__score_func': (mutual_info_classif, chi2)}
gs = GridSearchCV(rf, params, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(x_train, y_train)

print(f'Best score: {gs.best_score_}')
print(f'Best parameters: {gs.best_params_}')
```
Создана выборка с оптимальным количеством признаков:
```
selector = GenericUnivariateSelect(mode='k_best', score_func=chi2, param=59)
x_train_se = selector.fit_transform(x_train, y_train)
x_test_se = selector.transform(x_test)
```
Различные модели были проверены с помощью кросс-валидации:
```
scores = cross_val_score(rfc, x_train_se, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)
print('scores = {} \nmean score = {:.4f} +/- {:.4f}'.format(scores, scores.mean(), scores.std()))
```
Осуществлен подбор гиперпараметров лучших моделей:
```
params = {'n_estimators' : [300, 500, 700],
          'max_depth': np.arange(10, 60, 4),
          'min_samples_leaf': np.arange(1, 10, 1),
          'min_samples_split': np.arange(2, 20, 2)}

rs = RandomizedSearchCV(rfc, params, cv=kf, scoring='roc_auc', n_jobs=-1, error_score='raise')
rs.fit(x_train_se, y_train)

print('Best params: ', rs.best_params_)
print('Best score: ', rs.best_score_)
```
## Результат
Модель случайного леса показала лучший результат с оценкой ROC-AUC в 69%:

<img src="https://github.com/Dimayo/credit_scoring_project/assets/44707838/750655de-764e-4258-bde1-a41da9f8444a" width="600"> <br> <br>

Был создан пайплайн для обучения модели на всей выборке с предварительным отбором признаков, далее полученная модель была сохранена в pickle файл:
```
pipe = Pipeline([('selector', selector), ('classifier', rfc)])
pipe.fit(x, y)

joblib.dump(pipe, 'scoring_pipe.pkl')
```


