import matplotlib.pyplot as plt
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
    # Ver pie chart com a percentagem de depositos
    plt.pie(df['deposit'].value_counts(),labels=['no', 'yes'],autopct='%1.1f%%',  startangle=-90, colors=['lightcoral','lightgreen'])
    plt.title('Destribuição dos depositos')
    plt.show()

    # Order os meses antes de os apresentar
    ordem_meses = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    df['month'] = pd.Categorical(
        df['month'], 
        categories=ordem_meses, 
        ordered=True
    )

    # Ver graficos de barras a distribuiçao de depositos por todas as variaveis categoricas tendo em conta o deposito para cada categoria usando 2 barras
    plt.figure(figsize=(20,10))
    df.groupby(['job', 'deposit']).size().unstack().plot(
        kind='bar', 
        stacked=False, 
        ax=plt.gca(),
        color=['lightcoral','lightgreen'],
        edgecolor='black')
    plt.title('Distribution of deposits by job')
    plt.xlabel('job')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(20,10))
    i = 0
    for col in df.columns:
        if (df[col].dtype == 'object' and col not in ['deposit', 'job']) or col == 'month':  
            i += 1  
            plt.subplot(4, 2, i)
            df.groupby([col, 'deposit']).size().unstack().plot(
                kind='bar', 
                stacked=False, 
                ax=plt.gca(),
                color=['lightcoral','lightgreen'],
                edgecolor='black')
            plt.title(f'Distribution of deposits by {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

            
    i = 0
    plt.figure(figsize=(20,10))
    for col in df.columns:
        if df[col].dtype == 'int64':
            i += 1
            plt.subplot(4, 2, i)

            yes_data = df[df['deposit'] == 'yes'][col]
            no_data = df[df['deposit'] == 'no'][col]
            total_data = df[col]
            data_to_plot = [yes_data, no_data, total_data]
            colors = ['lightgreen', 'lightcoral', 'lightblue']
            boxprops = [dict(facecolor=color, color='blue') for color in colors]

            bp = plt.boxplot(data_to_plot, labels=['Yes', 'No', 'Total'], patch_artist=True, 
                    vert=False,
                    medianprops=dict(color='red'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    flierprops=dict(markerfacecolor='blue', marker='o', markersize=5, linestyle='none'))

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            plt.title(f'BoxPlot of {col}')
            plt.xlabel(col)
            plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def preposessing(df):
    # Preprocessamento
    dic = {"yes":1,"no":0}
    lst = ["deposit","loan","default","housing"]
    for i in lst:
        df[i] = df[i].map(dic).astype(int)

    l=['month',"contact","poutcome"]
    for i in l:
        le=LabelEncoder()
        df[i]=le.fit_transform(df[i].values)

    df = pd.get_dummies(df, columns = ['job','marital','education'])
    # Convert boolean columns to integers
    for column in df.columns:
        if df[column].dtype == 'bool':
            df[column] = df[column].astype(int)


    return df

def evaluate_classification_model(model, X_test, y_test, y_pred=None):
    if y_pred is None:
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
