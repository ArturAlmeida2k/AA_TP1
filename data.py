import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)



def vis(df):
    # Pie chart showing deposit percentage distribution
    plt.pie(df['deposit'].value_counts(), 
            labels=['no', 'yes'],
            autopct='%1.1f%%',  
            startangle=-90, 
            colors=['lightcoral','lightgreen'])
    plt.title('Deposit Distribution')
    plt.show()

    # Order months chronologically before visualization
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

    # Bar chart: deposit distribution by job
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

    # Bar charts: deposit distribution by each categorical variable
    plt.figure(figsize=(15, 9))
    i = 0
    for col in df.columns:
        if (df[col].dtype == 'object' and col not in ['deposit', 'job']) or col == 'month':
            i += 1
            plt.subplot(4, 2, i)
            df.groupby([col, 'deposit']).size().unstack().plot(
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


    # Histograms of numerical variables
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

    
    # Boxplots for numerical variables split by deposit
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



    # Correlation matrix heatmap
    df_temp = df.copy()
    df_temp['deposit'] = LabelEncoder().fit_transform(df_temp['deposit'])

    plt.figure(figsize=(15, 9))
    sns.heatmap(
        df_temp.select_dtypes(include=['number']).corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        linewidths=0.5
    )
    plt.title("Correlation Matrix of Numerical Variables", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def preposessing(df):
    # Aplicar pré-processamento
    dic = {"yes":1,"no":0}
    lst = ["deposit","loan","default","housing"]
    for i in lst:
        df[i] = df[i].map(dic).astype(int)

    # Education: primary < secondary < tertiary
    education_order = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": -1}
    df["education"] = df["education"].map(education_order).astype(int)

    # Month: Janeiro (1) a Dezembro (12)
    month_order = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df["month"] = df["month"].map(month_order).astype(int)

    nominal_cols = ["job", "marital", "contact", "poutcome"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)  # Evitar multicolinearidade

    # Convertendo colunas booleanas para inteiros
    for column in df.columns:
        if df[column].dtype == 'bool':
            df[column] = df[column].astype(int)

    return df

def evaluate_classification_model(model, X_test, y_test, y_pred=None):
    if y_pred is None:
    # Se não for passado y_pred, tentamos gerar a partir do modelo
        try:
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        except:
            y_pred = model.predict(X_test)
    
    # Matriz de Confusão (versão texto para terminal)
    cm = confusion_matrix(y_test, y_pred)
    print("╭────── Confusion Matrix ───────╮")
    print("│ Predicted:       No   |  Yes  │")
    print("├───────────────┬───────┬───────┤")
    print(f"│ Actual: No    │ {cm[0,0]:<6}│ {cm[0,1]:<6}│")
    print(f"│ Actual: Yes   │ {cm[1,0]:<6}│ {cm[1,1]:<6}│")
    print("╰───────────────┴───────┴───────╯")

    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nClassification Metrics:")
    print(f"Accuracy:  {accuracy*100:.4f} %")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Relatório completo
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
