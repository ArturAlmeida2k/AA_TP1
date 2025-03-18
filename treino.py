# Importações para a rede neural
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

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
    accuracy = np.mean(y_pred.flatten() == y_test.values) * 100
    print(f"Acurácia da rede neural: {accuracy:.2f}%")