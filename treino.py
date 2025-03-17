# Importações para a rede neural
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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