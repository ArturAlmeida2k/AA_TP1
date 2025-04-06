import pandas as pd
import data
import treino
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('Data/bank.csv')

# Initial data exploration
print("\n\n")
print("╔══════════════════════════════════════╗")
print("║         Dataset Preview             ║")
print("╚══════════════════════════════════════╝")
print(df.head())

print("\n╔══════════════════════════════════════╗")
print("║         Missing Values               ║")
print("╚══════════════════════════════════════╝")
print(df.isnull().sum())

print("\n╔══════════════════════════════════════╗")
print("║      Statistical Overview            ║")
print("╚══════════════════════════════════════╝")
print(df.describe())

print("\n╔══════════════════════════════════════╗")
print("║      Target Distribution             ║")
print("╚══════════════════════════════════════╝")
print(df['deposit'].value_counts())

print("\n╔══════════════════════════════════════╗")
print("║      Column Data Types               ║")
print("╚══════════════════════════════════════╝")
print(df.dtypes)

# Data visualization prompt
print("\n╔══════════════════════════════════════╗")
print("║  Show Visualizations? [Y/N]          ║")
print("╚══════════════════════════════════════╝")
if input().lower() in ['y', 'yes', 's', 'sim']:
    data.visualize_data(df)

# Data preprocessing
df = data.preprocess_data(df)

# Train-test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
for train_index, test_index in split.split(df, df['deposit']):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

# Distribution analysis
print("\n╔══════════════════════════════════════╗")
print("║    Train Set Distribution            ║")
print("╚══════════════════════════════════════╝")
print(train_set['deposit'].value_counts(normalize=True))

print("\n╔══════════════════════════════════════╗")
print("║     Test Set Distribution            ║")
print("╚══════════════════════════════════════╝")
print(test_set['deposit'].value_counts(normalize=True))

# Prepare features and target
X_train = train_set.drop(columns=['deposit'])
y_train = train_set['deposit']
X_test = test_set.drop(columns=['deposit'])
y_test = test_set['deposit']

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model selection interface
print("\n════════════════════════════════════ Model Testing ═════════════════════════════════════\n")
while True:
    print("Choose a model:")
    print("1 - Logistic Regression ")
    print("2 - Neural Network ")
    print("3 - Support Vector Machine ")
    print("4 - Get best SVM parameters")
    print("5 - Get best NN parameters")
    print("0 - Exit")

    choice = input("Option: ").strip()

    match choice:
        case "1": #(≈69% acc)
            print("\nRunning Logistic Regression")
            model, predictions = treino.logistic_regression(X_train, y_train, X_test, y_test)
            data.evaluate_model_performance(model, X_test, y_test, predictions)

        case "2": #(≈71% acc)
            print("\nTraining Neural Network")
            model, predictions = treino.neural_network(X_train, y_train, X_test, y_test)
            data.evaluate_model_performance(model, X_test, y_test, predictions)

        case "3": #(≈70% acc)
            print("\nRunning SVM Classifier")
            model, predictions = treino.svm_classifier(X_train, y_train, X_test, y_test)
            data.evaluate_model_performance(model, X_test, y_test, predictions)

        case "4": 
            print("\nRunning SVM Optimizer")
            model = treino.optimized_svm(X_train, y_train, X_test, y_test)

        case "5": 
            print("\nRunning NN Optimizer")
            model = treino.dynamic_nn_search(X_train, y_train)

        case "0":
            print("Exiting program...")
            break
    
        case _:
            print("\nInvalid option. Please choose 1-5.")