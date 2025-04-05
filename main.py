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

print("\n\n")
print("╔══════════════════════════════════════╗")
print("║       FIRST ROWS OF THE DATASET      ║")
print("╚══════════════════════════════════════╝")
print(df.head())

print("\n╔══════════════════════════════════════╗")
print("║         MISSING VALUES CHECK         ║")
print("╚══════════════════════════════════════╝")
print(df.isnull().sum())

print("\n╔══════════════════════════════════════╗")
print("║        DESCRIPTIVE STATISTICS        ║")
print("╚══════════════════════════════════════╝")
print(df.describe())

print("\n╔══════════════════════════════════════╗")
print("║      TARGET VARIABLE DISTRIBUTION    ║")
print("╚══════════════════════════════════════╝")
print(df['deposit'].value_counts())

print("\n╔══════════════════════════════════════╗")
print("║       DATA TYPES PER COLUMN          ║")
print("╚══════════════════════════════════════╝")
print(df.dtypes)


# Data Visualization

print("\n╔══════════════════════════════════════╗")
print("║  DO YOU WANT TO VIEW THE DATA? [Y/N] ║")
print("╚══════════════════════════════════════╝")
if input().lower() in ['y', 'yes', 's', 'sim']:
    data.vis(df)

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

print("\n╔══════════════════════════════════════╗")
print("║    DISTRIBUIÇÃO TARGET NO TREINO     ║")
print("╚══════════════════════════════════════╝")
print(train['deposit'].value_counts() / train.shape[0])

print("\n╔══════════════════════════════════════╗")
print("║     DISTRIBUIÇÃO TARGET NO TESTE     ║")
print("╚══════════════════════════════════════╝")
print(test['deposit'].value_counts() / test.shape[0])



X_train = train.drop(columns=['deposit'])
y_train = train['deposit']

X_test = test.drop(columns=['deposit'])
y_test = test['deposit']

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#        -------------------------------- Tests ------------------------------------
print("\n════════════════════════════════════ Tests ═════════════════════════════════════\n")
while True:
    print("Escolha um modelo:")
    print("1 - Logistic Regression")
    print("2 - Neural Networks")
    print("3 - SVM")
    print("4 - Exit")

    opcao = input("Opção: ").strip()

    match opcao:
        case "1": # (~80.30%)
            print("\nExecutando Regressão Logística")
            logreg_model, y_pred_logreg = treino.regressao_logistica(X_train, y_train, X_test, y_test)
            data.evaluate_classification_model(logreg_model, X_test, y_test, y_pred_logreg, "Logist Regression")

        case "2": # (~82.17%)
            print("\nExecutando Rede Neural")
            model_rn, y_pred_rn = treino.neural_network(X_train, y_train, X_test, y_test)
            data.evaluate_classification_model(model_rn, X_test, y_test, y_pred_rn, "Neural Network")

        case "3": # (~80.30%)
            print("\nExecutando SVM")
            y_pred_svm = treino.svm_model(X_train, y_train, X_test, y_test)
            data.evaluate_classification_model(None, X_test, y_test, y_pred_svm, "SVM")

        case "4":
            print
            break
        
        case _:
            print("\nOpção inválida. Por favor escolha 1, 2 ou 3.")
