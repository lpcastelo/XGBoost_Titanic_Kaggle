# XGBoost_Titanic_Kaggle
Aqui você irá encontrar a atividade do módulo 41 do curso de Cientista de Dados da Ebac.

A atividade consiste em desenvolver um modelo de Machine Learning utilizando o algoritmo XGBoost para a famosa competição **Titanic: Machine Learning from Disaster**, do Kaggle, cujo objetivo é prever quais passageiros sobreviveram ao naufrágio do Titanic.

Durante a atividade serão abordados temas como:

* Análise exploratória de dados (EDA)
* Engenharia e pré-processamento de atributos
* Otimização de hiperparâmetros com GridSearchCV
* Validação cruzada (Cross Validation)
* Balanceamento de classes
* Padronização e normalização dos dados
* Treinamento e avaliação do modelo XGBoost
* Geração de previsões e submissão no Kaggle

O objetivo é consolidar os conhecimentos adquiridos ao longo do curso, aplicando diversas técnicas de preparação de dados e otimização de modelos em um problema real de Ciência de Dados, além de avaliar o desempenho do modelo por meio de uma competição amplamente utilizada pela comunidade de Machine Learning.

## Resultados

O tratamento dos dados incluiu imputação de idade/tarifa pela mediana, transformação de `Cabin` em indicador binário, remoção de `Name`/`Ticket`, codificação de `Sex` e one-hot encoding de `Embarked`, além de balanceamento com SMOTE e padronização (`StandardScaler`).

Um primeiro modelo XGBoost com `GridSearchCV` (otimizado por acurácia de treino) atingiu **91,9% de acurácia no treino**, mas apenas **69,4% no Kaggle** — um caso claro de **overfitting**. Para corrigir isso, o modelo foi retreinado com regularização mais forte (`reg_alpha`/`reg_lambda`, menor profundidade de árvore, subsample), o que reduziu a acurácia de treino (para ~82%) mas **melhorou o resultado real no Kaggle para 0,74401**, confirmando a hipótese e demonstrando na prática o trade-off entre ajuste ao treino e capacidade de generalização.

## Tecnologias

- Python, pandas, numpy
- scikit-learn (GridSearchCV, StandardScaler)
- XGBoost
- imbalanced-learn (SMOTE)

## Como executar

1. Instale as dependências: `pip install pandas numpy scikit-learn xgboost imbalanced-learn`.
2. Coloque `train.csv` e `test.csv` (dados da competição Titanic do Kaggle) no mesmo diretório do notebook.
3. Execute `Profissao Cientista de Dados M41 Projeto.ipynb` em ordem. As previsões finais são salvas em `.csv`, prontas para submissão no Kaggle.
