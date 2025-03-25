import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data
import treino

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import metrics #accuracy measure


df = pd.read_csv('Data/bank.csv')

print(df.head())

print(df.isnull().sum())
print(df.describe())
print(df['deposit'].value_counts())
print(df.dtypes)

# Visualização dos dados
# data.vis(df)

# Preprocessamento dos dados
df = data.preposessing(df)

# print(df)
# print(df.describe())
# df.to_csv('AA_TP1/Data/bank_preprocessed.csv', index=False)

# Divisão dos dados em treino e teste
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
for train_index, test_index in split.split(df, df['deposit']):
    train = df.loc[train_index]
    test = df.loc[test_index]

print("Ratio for train dataset")
print(train['deposit'].value_counts()/train.shape[0])
print()
print("ratio for test dataset")
print(test['deposit'].value_counts()/test.shape[0])

X_train = train.drop(columns=['deposit'])
y_train = train['deposit']

X_test = test.drop(columns=['deposit'])
y_test = test['deposit']

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#        -------------------------------- Tests ------------------------------------
print("\n-------------------------------- Tests-------------------------------------\n")
print("Escolhe um modelo:")
print("1 - Logistic Regression")
print("2 - Neural Networks")
print("3 - Neural Networks 2")
print("4 - SVM")

opcao = input("Opção: ").strip()

match opcao:
    case "1": # (~80.30%)
        print("\nExecutando Regressão Logística")
        logreg_model, y_pred_logreg = treino.regressao_logistica(X_train, y_train, X_test, y_test)
        data.evaluate_classification_model(logreg_model, X_test, y_test, y_pred_logreg)

    case "2": # (~80.92%)
        print("\nExecutando Rede Neural 1")
        model_rn, y_pred_rn = treino.rede_neural(X_train, y_train, X_test, y_test)
        data.evaluate_classification_model(model_rn, X_test, y_test, y_pred_rn)

    case "3": # (~79.49%)
        print("\nExecutando Rede Neural 2")
        y_pred = treino.rede_neural2(X_train, y_train, X_test, y_test)
        data.evaluate_classification_model(model=None, X_test=X_test, y_test=y_test, y_pred=y_pred)

    case "4": # (~80.30%)
        print("\nExecutando SVM")
        y_pred_svm = treino.svm_model(X_train, y_train, X_test, y_test)
        data.evaluate_classification_model(model=None, X_test=X_test, y_test=y_test, y_pred=y_pred_svm)

    case _:
        print("\nOpção inválida. Por favor escolha 1, 2, 3 ou 4.")
