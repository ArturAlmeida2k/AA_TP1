# Importações para a rede neural
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import RandomizedSearchCV

# Caso necessário, converta o target para formato numérico (0/1) se ainda não estiver
# y_train = y_train.map({'no': 0, 'yes': 1})
# y_test = y_test.map({'no': 0, 'yes': 1})
def rede_neural(X_train, y_train, X_test, y_test):
    # Definição da arquitetura do modelo
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1], )),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Ajuste para problemas binários
    ])

    # Compilação do modelo
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Treinamento do modelo
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=32, 
                        validation_split=0.2)

    # Avaliação do modelo no conjunto de teste
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Precisão da rede neural no conjunto de teste: {accuracy:.2f}")

    # (Opcional) Visualização dos resultados do treinamento
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='acurácia treinamento')
    plt.plot(history.history['val_accuracy'], label='acurácia validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()


# Função de ativação sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Função de previsão para a rede neural
def predict(Theta1, Theta2, X):
    """Realiza previsão usando uma rede neural de duas camadas"""
    m = X.shape[0]
    
    # Adiciona uma coluna de bias (1s) na entrada
    X = np.append(np.ones((m, 1)), X, axis=1)
    
    # Forward propagation - Camada oculta
    z1 = np.dot(X, Theta1.T)
    a1 = sigmoid(z1)
    
    # Adiciona coluna de bias na camada oculta
    a1 = np.append(np.ones((m, 1)), a1, axis=1)

    # Forward propagation - Camada de saída
    z2 = np.dot(a1, Theta2.T)
    a2 = sigmoid(z2)

    return (a2 >= 0.5).astype(int)  # Ajustado para saída binária

def rede_neural2(X_train, y_train, X_test, y_test):
    # Inicialização aleatória dos pesos da rede
    print(X_train.shape[1])
    input_layer_size = X_train.shape[1]  # Número de features
    hidden_layer_size = 25  # Número arbitrário de neurônios na camada oculta
    output_layer_size = 1  # Saída binária

    # Pesos aleatórios entre -0.5 e 0.5
    Theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1) - 0.5
    Theta2 = np.random.rand(output_layer_size, hidden_layer_size + 1) - 0.5

    # Treinamento simples usando descida do gradiente
    alpha = 0.01  # Taxa de aprendizado
    epochs = 10000

    for i in range(epochs):
        # Forward propagation
        m = X_train.shape[0]
        X_bias = np.append(np.ones((m, 1)), X_train, axis=1)
        z1 = np.dot(X_bias, Theta1.T)
        a1 = sigmoid(z1)
        a1_bias = np.append(np.ones((m, 1)), a1, axis=1)
        z2 = np.dot(a1_bias, Theta2.T)
        a2 = sigmoid(z2)

        # Cálculo do erro
        error = a2 - y_train.to_numpy().reshape(-1, 1)

        # Backpropagation - Ajuste dos pesos
        dTheta2 = np.dot(error.T, a1_bias) / m
        dTheta1 = np.dot(((error @ Theta2[:,1:]) * a1 * (1 - a1)).T, X_bias) / m

        # Atualização dos pesos
        Theta1 -= alpha * dTheta1
        Theta2 -= alpha * dTheta2

        if i % 100 == 0:
            y_train_reshaped = y_train.to_numpy().reshape(-1, 1)  # Converter para array 2D
            loss = np.mean(-y_train_reshaped * np.log(a2) - (1 - y_train_reshaped) * np.log(1 - a2))
            print(f"Época {i}: Loss = {loss:.4f}")

    # Avaliação do modelo
    y_pred = predict(Theta1, Theta2, X_test)
    #accuracy = np.mean(y_pred.flatten() == y_test.values) * 100
    #print(f"Acurácia da rede neural: {accuracy:.2f}%")
    return y_pred
    
def regressao_logistica(X_train, y_train, X_test, y_test):

    # Criar e treinar o modelo de Regressão Logística
    log_reg = LogisticRegression(max_iter=1000)  # max_iter para garantir convergência
    log_reg.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = log_reg.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Exibir resultados
    print(f"Acurácia da Regressão Logística: {accuracy:.2f}")
    print("\nRelatório de Classificação:")
    print(report)
    print("\nMatriz de Confusão:")
    print(cm)

    return accuracy, report, cm



# Função para criar a rede neural
def create_model(input_dim, hidden_size=50, learning_rate=0.001, lambda_reg=0.01):
    model = Sequential([
        Dense(hidden_size, activation='relu', kernel_regularizer=l2(lambda_reg), input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Função principal
def rede_neural_dinamica(df):
    # Divisão treino/teste (80/20)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
    for train_index, test_index in split.split(df, df['deposit']):
        train, test = df.iloc[train_index], df.iloc[test_index]

    # Pré-processamento
    X_train, y_train = train.drop(columns=['deposit']).values, train['deposit'].values
    X_test, y_test = test.drop(columns=['deposit']).values, test['deposit'].values

    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

    input_dim = X_train.shape[1]

    # Hiperparâmetros para testar manualmente
    hidden_sizes = np.arange(10, 100, 20)  # Testa valores entre 10 e 90
    learning_rates = [1e-4, 1e-3, 1e-2]
    lambda_regs = [0.001, 0.01, 0.1, 1.0]

    best_acc = 0
    best_model = None
    best_params = {}

    # Cross-validation manual
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for lambda_reg in lambda_regs:
                print(f"Testando: hidden_size={hidden_size}, learning_rate={learning_rate}, lambda_reg={lambda_reg}")

                model = create_model(input_dim, hidden_size, learning_rate, lambda_reg)

                # Cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
                val_accs = []

                for train_idx, val_idx in cv.split(X_train_scaled, y_train):
                    X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

                    model.fit(X_train_cv, y_train_cv, epochs=100, verbose=0, batch_size=32)
                    val_acc = model.evaluate(X_val_cv, y_val_cv, verbose=0)[1]
                    val_accs.append(val_acc)

                mean_acc = np.mean(val_accs)
                print(mean_acc)
                # Verificar se é o melhor modelo até agora
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_model = model
                    best_params = {
                        "hidden_size": hidden_size,
                        "learning_rate": learning_rate,
                        "lambda_reg": lambda_reg
                    }

    print("Melhores hiperparâmetros:", best_params)
    print(f"Melhor acurácia de validação: {best_acc:.2%}")

    # Avaliação no conjunto de teste
    test_acc = best_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
    print(f"Acurácia no teste: {test_acc:.2%}")

    return best_model