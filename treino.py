from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt


def logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate a logistic regression model"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def svm_classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate a baseline SVM classifier"""
    model = SVC(C=1, gamma='scale', kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

def optimized_svm(X_train, y_train, X_test, y_test):
    """Perform hyperparameter optimization for SVM"""
    timer = {
        'setup': time.time(),
        'search': None,
    }
    
    model = SVC(probability=True, random_state=42, class_weight='balanced')
    
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    timer['setup'] = time.time() - timer['setup']
    print(f"[SVM] Parameter setup completed in {timer['setup']:.2f}s")

    # Hyperparameter search
    timer['search'] = time.time()
    search = GridSearchCV(
        model, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    timer['search'] = time.time() - timer['search']

    # Results summary
    print(f"Optimal parameters: {search.best_params_}")
    print(f"Cross-validation accuracy: {search.best_score_:.4f}")

    return search.best_estimator_

def neural_network(X_train, y_train, X_test, y_test):
    """Train and evaluate a multi-layer perceptron"""
    architecture = (10, 10, 20)
    hyperparams = {
        'learning_rate': 0.01,
        'regularization': 0.01,
        'max_epochs': 1000
    }
    
    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        learning_rate_init=hyperparams['learning_rate'],
        alpha=hyperparams['regularization'],
        max_iter=hyperparams['max_epochs'],
        early_stopping=True,
        random_state=42,
        verbose=False
    )

    model.fit(X_train, y_train)
    
    # Model evaluation
    predictions = model.predict(X_test)

    # Training visualization
    plt.figure(figsize=(10, 4))
    plt.plot(model.loss_curve_, linewidth=2)
    plt.title("Training Loss Progression")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return model, predictions

def dynamic_nn_search(X_train, y_train):
    """Perform architectural search for neural networks"""
    param_grid = {
        'hidden_layer_sizes': [
            (h1, h2, h3) 
            for h1 in range(10, 110, 30) 
            for h2 in range(10, 60, 15) 
            for h3 in range(5, 25, 5)
        ],
        'alpha': [0.001, 0.01, 0.1],
        'learning_rate_init': [0.0001, 0.001, 0.01]
    }

    model = MLPClassifier(
        max_iter=1000,
        early_stopping=True,
        random_state=42,
        verbose=False
    )

    search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)

    print("\nOptimal parameters found:", search.best_params_)
    print(f"Cross-validation accuracy: {search.best_score_:.2%}")

    return search.best_estimator_