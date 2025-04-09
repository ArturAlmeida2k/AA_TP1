import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve, 
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    average_precision_score
)

# Set backend for matplotlib
plt.switch_backend('TkAgg')

def visualize_data(df):
    """Generate comprehensive data visualizations"""
    # Target distribution pie chart
    plt.pie(df['deposit'].value_counts(), 
            labels=['No', 'Yes'],
            autopct='%1.1f%%',  
            startangle=-90, 
            colors=['lightcoral', 'lightgreen'])
    plt.title('Subscription Distribution')
    plt.show()

    # Temporal ordering for months
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

    # Categorical feature analysis
    plot_categorical_distributions(df)
    plot_numerical_distributions(df)
    generate_correlation_heatmap(df)

def preprocess_data(df):    
    """Clean and transform raw data into machine-readable format"""
    # Remove non-predictive feature
    df = df.drop(columns=['duration'])
    
    # Binary feature encoding
    binary_map = {"yes": 1, "no": 0}
    binary_features = ["deposit", "loan", "default", "housing"]
    for feature in binary_features:
        df[feature] = df[feature].map(binary_map).astype('int8') 

    # Ordinal encoding for education levels
    education_levels = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": -1}
    df["education"] = df["education"].map(education_levels).astype('int8')

    # Temporal feature encoding
    month_order = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df["month"] = df["month"].map(month_order).astype(int)

    # Nominal feature encoding
    categorical_features = ["job", "marital", "contact", "poutcome"]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)

    return df

def preprocess_data2(df):    
    """Clean and transform raw data into machine-readable format"""
    # Remove non-predictive feature
    df = df.drop(columns=['duration', 'job'])
    
    # Binary feature encoding
    binary_map = {"yes": 1, "no": 0}
    binary_features = ["deposit", "loan", "default", "housing"]
    for feature in binary_features:
        df[feature] = df[feature].map(binary_map).astype('int8') 

    # Ordinal encoding for education levels
    education_levels = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": -1}
    df["education"] = df["education"].map(education_levels).astype('int8')

    # Temporal feature encoding
    month_order = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df["month"] = df["month"].map(month_order).astype(int)

    # Nominal feature encoding
    categorical_features = ["marital", "contact", "poutcome"]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)

    return df

def evaluate_model_performance(model, X_test, y_test, predictions):
    """Generate comprehensive model evaluation metrics and visualizations"""
    
    print("\nClassification Metrics:\n")
    print(classification_report(y_test, predictions, digits=4))

    generate_precision_recall_curve(model, X_test, y_test)

    cm = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes']).plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()
    
    generate_roc_curve(model, X_test, y_test)

# Helper functions ------------------------------------------------------------

def plot_categorical_distributions(df):
    """Visualize distributions of categorical features"""
    plt.figure(figsize=(15, 9))
    df.groupby(['job', 'deposit']).size().unstack().plot(
        kind='bar',
        stacked=False, 
        ax=plt.gca(),
        color=['lightcoral', 'lightgreen'], 
        edgecolor='black')
    plt.title('Deposit Distribution by Job')
    plt.xlabel('Job')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 9))
    i = 0
    for col in df.columns:
        if (df[col].dtype == 'object' and col not in ['deposit', 'job']) or col == 'month':
            i += 1
            plt.subplot(4, 2, i)
            df.groupby([col, 'deposit'], observed=False).size().unstack().plot(
                kind='bar', 
                stacked=False, 
                ax=plt.gca(),
                color=['lightcoral', 'lightgreen'], 
                edgecolor='black')
            plt.title(f'Deposit Distribution by {col.capitalize()}')
            plt.xlabel(col.capitalize())
            plt.ylabel('Count')
            plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_numerical_distributions(df):
    """Visualize distributions of numerical features"""    
    plt.figure(figsize=(15, 9))
    i = 0
    for col in df.columns:
        if df[col].dtype == 'int64':
            i += 1
            plt.subplot(4, 2, i)
            sns.histplot(df[col], bins=20, color='lightblue')
            plt.title(f'Distribution of {col}', fontsize=12)
            plt.xlabel(col)
            plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 9))
    i = 0
    for col in df.columns:
        if df[col].dtype == 'int64':
            i += 1
            plt.subplot(4, 2, i)
            yes_data = df[df['deposit'] == 'yes'][col]
            no_data = df[df['deposit'] == 'no'][col]
            total_data = df[col]

            data_to_plot = [yes_data, no_data, total_data]

            bp = plt.boxplot(data_to_plot,
                            labels=['Yes', 'No', 'Total'], 
                            patch_artist=True, 
                            vert=False,
                            medianprops=dict(color='red'),
                            whiskerprops=dict(color='blue'),
                            capprops=dict(color='blue'),
                            flierprops=dict(markerfacecolor='blue', marker='o', markersize=5, linestyle='none'))

            colors = ['lightgreen', 'lightcoral', 'lightblue']

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
            plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_correlation_heatmap(df):
    """Generate feature correlation matrix visualization"""
    df_temp = df.copy()
    
    if df_temp['deposit'].dtype == 'object':
        df_temp['deposit'] = LabelEncoder().fit_transform(df_temp['deposit'])
    
    categorical_cols = df_temp.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col == 'month':
            month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            df_temp[col] = df_temp[col].astype('category').cat.reorder_categories(month_order).cat.codes + 1

        elif set(df_temp[col].unique()) == {'yes', 'no'}:
            df_temp[col] = df_temp[col].map({'yes': 1, 'no': 0})

        else:
            df_temp[col] = LabelEncoder().fit_transform(df_temp[col])
    
    numeric_df = df_temp.select_dtypes(include=['number'])
    
    plt.figure(figsize=(18, 12))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        mask=np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    )
    
    plt.title("Correlation Matrix Including All Encoded Variables", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

def generate_roc_curve(model, X_test, y_test):
    """Generate ROC curve visualization"""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def generate_precision_recall_curve(model, X_test, y_test):
    """Gera a curva Precision-Recall"""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    precision, recall, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)

    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'AP = {ap:.2f}', color='darkorange')
    plt.title('Precision-Recall Relationship')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()
