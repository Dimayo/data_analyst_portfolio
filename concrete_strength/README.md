# Прогноз прочности бетона
[Код проекта в jupyter notebook](https://github.com/Dimayo/concrete_strength/blob/main/concrete.ipynb)<br>
Библиотеки python: pandas, seaborn, matplotlib, numpy, sklearn

## Цель проекта
<p>Целью данного проекта было выявление зависимости прочности бетона от его состава и технологии производства, используя результаты проведенных испытаний. Необходимо было построить модель машинного обучения, реализующую данный подход.</p> <p>Успешное решение подобных задач позволяет существенно снизить затраты и ускорить производство качественного продукта.</p>

## Описание проекта
<p>В качестве данных даны два файла, содержащие обучающую и тестовую выборку. Обучающая выборка состоит из десяти столбцов. Первый — это идентификатор состава (точки данных), следующие восемь — независимые переменные, а девятый (последний) — это цель, которую должна предсказать модель.</p> <p>Тестовая выборка состоит из девяти столбцов — такиж же как и в обучающей выборке, но без целевой переменной.</p>
<p>Решения оцениваются по значению корня среднеквадратической ошибки.</p>

## Что было сделано
Данные были загружены, была произведена проверка данных на пропуски и удаление дубликатов. Далее была произведена проверка зависимости целевой переменной от признаков с помощью метода pairplot библиотеки seaborn, также была проверена информация о типах данных и описательная статистика датафрейма. Проверены выбросы в признаках:

```
for column in df.columns:
    plt.figure()
    sns.boxplot(df[column])
    plt.title(column)
```
Выбросы были заменены граничным значением с помощью функции для рассчета интерквартильного размаха:
```
for column in x.columns:
    x.loc[x[column] > get_outliers(x[column])[1], column] = get_outliers(x[column])[1]
    x.loc[x[column] < get_outliers(x[column])[0], column] = get_outliers(x[column])[0]
    x_test.loc[x_test[column] > get_outliers(x[column])[1], column] = get_outliers(x[column])[1]
    x_test.loc[x_test[column] < get_outliers(x[column])[0], column] = get_outliers(x[column])[0]
```
Выборки были стандартизированы:
```
scaler = StandardScaler()
columns = x.columns
x[columns] = scaler.fit_transform(x[columns])
x_test[columns] = scaler.transform(x_test[columns])
```
Были протестированы различные модели с подбором гиперпараметров с помщью Grid Search:
```
params = {
    'hidden_layer_sizes': [(200,150,100),(500,500,500)],
    'max_iter': [2000,2500,3000],
    'activation': ['relu','logistic'],
    'solver': ['sgd','adam'],
    'alpha': [1,10],
    'learning_rate': ['adaptive','constant'],
}

gs = GridSearchCV(MLPRegressor(), params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
gs.fit(x, y)

print('Лучшие параметры для MLPRegressor: ', gs.best_params_)
print('Лучший результат для MLPRegressor: ', gs.best_score_)
```
## Результат
Лучший результат показала модель многослойного персептрона с корнем среднеквадратической ошибки равным 4.9:
```
model = MLPRegressor(activation='relu', hidden_layer_sizes=(600, 550, 500), alpha=10,
                     learning_rate='adaptive', max_iter=2000, solver='sgd')
model.fit(x, y)
```
Создан файл, который будет использоваться для проверки качества модели на тестовой выборке:
```
df_submission = pd.DataFrame(data= {
    'Id': df_test['Id'],
    'Strength': test_pred
})
df_submission.head()
```
