# Importações para a rede neural
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,  roc_curve, auc
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt

def regressao_logistica(X_train, y_train, X_test, y_test):

    # Criar e treinar o modelo de Regressão Logística
    log_reg = LogisticRegression(max_iter=1000)  # max_iter para garantir convergência
    log_reg.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = log_reg.predict(X_test)

    return log_reg, y_pred


def svm_model(X_train, y_train, X_test, y_test):

    # 1. Configuração do modelo e parâmetros
    start_time = time.time()
    
    svm = SVC(probability=True, random_state=5, class_weight='balanced')
    
    # Espaço de busca para os hiperparâmetros
    param_grid = {
        'C': [0.1, 1, 10],           # Valores típicos em escala logarítmica
        'kernel': ['linear', 'rbf'],   # Kernels mais comuns
        'gamma': ['scale']             # Evita testar múltiplos gammas
    }
    
    setup_time = time.time() - start_time
    print(f"[SVM] Configuração inicial: {setup_time:.2f}s")

    # 2. Busca em grade 
    start_time = time.time()
    
    # Opção 1: Usar todos os dados (mais lento)
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1  # Paraleliza usando todos os núcleos
    )
    grid_search.fit(X_train, y_train)
    
    search_time = time.time() - start_time
    print(f"[SVM] Busca em grade: {search_time:.2f}s")

    # 3. Avaliação do melhor modelo
    start_time = time.time()
    
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    
    eval_time = time.time() - start_time
    print(f"[SVM] Predição no teste: {eval_time:.2f}s")

    # 4. Relatório completo
    print("\n" + "="*50)
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Acurácia do SVM: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Tempo total: {setup_time + search_time + eval_time:.2f}s")
    print("="*50 + "\n")
    
    return y_pred

def neural_network(X_train, y_train, X_test, y_test):
    # Definir e treinar o modelo
    hidden_layers = (10, 10, 25)
    learning_rate = 0.01
    lambda_reg = 0.1

    print(f"Treinando modelo com hidden_layers={hidden_layers}, learning_rate={learning_rate}, lambda_reg={lambda_reg}")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=learning_rate,
        alpha=lambda_reg,
        max_iter=100,
        batch_size=32,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
        # verbose=True
    )

    model.fit(X_train, y_train)

    # Avaliação
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia no teste: {acc:.2%}")

    # # Plotar a função de custo (loss)
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(model.loss_curve_)
    # plt.title("Função de Custo (Loss) ao longo das épocas")
    # plt.xlabel("Épocas")
    # plt.ylabel("Loss")
    # plt.grid(True)

    # # Calcular e plotar a curva ROC
    # y_score = model.predict_proba(X_test)[:, 1]
    # fpr, tpr, thresholds = roc_curve(y_test, y_score)
    # roc_auc = auc(fpr, tpr)

    # plt.subplot(1, 2, 2)
    # plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Curva ROC')
    # plt.legend(loc='lower right')
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    return model, y_pred    


def rede_neural_dinamica2(X_train, y_train, X_test, y_test):

    # Parâmetros para busca
    param_grid = {
        'hidden_layer_sizes': [(h1, h2, h3) for h1 in range(10, 110, 20) for h2 in range(10, 110, 20) for h3 in range(5, 45, 5)],
        'alpha': [0.001, 0.01, 0.1], 
        'learning_rate_init': [0.0001, 0.001, 0.01]
    }

    # Modelo de rede neural com sklearn
    model = MLPClassifier(max_iter=1000, random_state=42)

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    test_acc = best_model.score(X_test, y_test)
    print(f"Acurácia no teste: {test_acc:.2%}")

    return best_model