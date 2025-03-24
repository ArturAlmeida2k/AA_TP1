# Importações para a rede neural
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
<<<<<<< HEAD

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


def rede_neural3(df):
    # Divisão inicial: 80% treino, 20% teste
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
    for train_index, test_index in split.split(df, df['deposit']):
        train = df.loc[train_index]
        test = df.loc[test_index]

    # Separar features (X) e target (y)
    X_train = train.drop(columns=['deposit'])
    y_train = train['deposit']
    X_test = test.drop(columns=['deposit'])
    y_test = test['deposit']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Ajusta e transforma o treino
    X_test_scaled = scaler.transform(X_test)        # Apenas transforma o teste

    # Hiperparâmetros para otimização
    hidden_sizes = [20, 50]         # Testar diferentes tamanhos
    alphas = [0.01, 0.001]          # Taxas de aprendizado
    lambda_regs = [0.01, 0.1]       # Valores de regularização
    best_accuracy = 0
    best_params = {}
    
    # Validação cruzada para otimização
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    
    for hidden_size in hidden_sizes:
        for alpha in alphas:
            for lambda_reg in lambda_regs:
                print(f"\nTestando: hidden={hidden_size}, alpha={alpha}, lambda={lambda_reg}")
                fold_accuracies = []
                
                for train_idx, val_idx in kf.split(X_train, y_train):
                    # Divisão dos dados
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    
                    # Inicialização Xavier/Glorot
                    input_size = X_tr.shape[1]
                    Theta1 = np.random.randn(hidden_size, input_size + 1) * np.sqrt(2/(input_size + hidden_size))
                    Theta2 = np.random.randn(1, hidden_size + 1) * np.sqrt(2/(hidden_size + 1))
                    
                    # Treino com early stopping
                    best_val_loss = float('inf')
                    patience = 20
                    wait = 0
                    
                    for epoch in range(1000):
                        # Forward propagation
                        X_bias = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
                        z1 = X_bias @ Theta1.T
                        a1 = relu(z1)
                        a1_bias = np.hstack([np.ones((a1.shape[0], 1)), a1])
                        z2 = a1_bias @ Theta2.T
                        a2 = sigmoid(z2)
                        
                        # Backpropagation com regularização
                        error = a2 - y_tr.reshape(-1,1)
                        delta2 = error.T @ a1_bias / X_tr.shape[0]
                        delta1 = (error @ Theta2[:,1:] * (z1 > 0)) @ X_bias / X_tr.shape[0]
                        
                        # Regularização L2
                        delta2 += (lambda_reg/X_tr.shape[0]) * np.hstack([np.zeros((Theta2.shape[0],1)), Theta2[:,1:]])
                        delta1 += (lambda_reg/X_tr.shape[0]) * np.hstack([np.zeros((Theta1.shape[0],1)), Theta1[:,1:]])
                        
                        # Atualização de pesos
                        Theta2 -= alpha * delta2
                        Theta1 -= alpha * delta1
                        
                        # Early stopping
                        val_pred = predict(Theta1, Theta2, X_val)
                        val_acc = accuracy_score(y_val, val_pred)
                        
                        if val_acc > best_val_loss:
                            best_val_loss = val_acc
                            wait = 0
                        else:
                            wait += 1
                            if wait >= patience:
                                break
                                
                    fold_accuracies.append(val_acc)
                
                avg_acc = np.mean(fold_accuracies)
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                    best_params = {
                        'hidden_size': hidden_size,
                        'alpha': alpha,
                        'lambda_reg': lambda_reg
                    }
    
    print(f"\nMelhores parâmetros: {best_params}")
    
    # Treino final com todos os dados de treino
    input_size = X_train.shape[1]
    Theta1 = np.random.randn(best_params['hidden_size'], input_size + 1) * np.sqrt(2/(input_size + best_params['hidden_size']))
    Theta2 = np.random.randn(1, best_params['hidden_size'] + 1) * np.sqrt(2/(best_params['hidden_size'] + 1))
    
    for epoch in range(1000):
        # ... (mesma lógica de treino com early stopping)
    
    # Avaliação final
    y_pred = predict(Theta1, Theta2, X_test)
    final_acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia final no teste: {final_acc:.2%}")
    
    return Theta1, Theta2


# Função de custo com regularização L2
def compute_cost(a2, y, Theta1, Theta2, lambda_reg):
    m = y.shape[0]
    cost = -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))
    reg = (lambda_reg/(2*m)) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
    return cost + reg

=======
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model  # Retorna apenas o modelo treinado
>>>>>>> f985f2825adb4c8bf9fe26ea31bffbf9578f3719
