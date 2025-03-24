# %% [markdown]
# # 📥 Importação das Bibliotecas

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data
import treino
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import metrics #accuracy measure
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


# %% [markdown]
# # 📊 Carregar e Inspecionar os Dados

# %%
# Carregar os dados
df = pd.read_csv('Data/bank.csv')
#df = pd.read_csv('/home/rafa/Secretária/AA/AA_TP1/Data/bank.csv')

# Exibir as primeiras linhas
print(df.head())

correlation_matrix = df.select_dtypes(include=['number']).corr()
print(correlation_matrix)

# %%
# Verificar valores ausentes
print(df.isnull().sum())

# %%
# Estatísticas básicas
print(df.describe())

# %%
# Distribuição da variável alvo
print(df['deposit'].value_counts())

# %%
# Tipos de dados
print(df.dtypes)

# %%
for col in df.columns:
    if df[col].dtype == "object":
        print(f"\n=== Coluna: {col} ===")
        print(f"Tipo de dados: {df[col].dtype}")
        print("Valores únicos:")
        
        # Se for uma coluna categórica ou com poucos valores únicos, mostra todos
        
        print(df[col].value_counts(dropna=False))  # dropna=False inclui NaN
        

# %% [markdown]
# # 📊 Visualização dos Dados

# %%
# Pie Chart da variável alvo (depósitos)
plt.pie(df['deposit'].value_counts(), labels=['no', 'yes'], autopct='%1.1f%%', startangle=-90, colors=['lightcoral', 'lightgreen'])
plt.title('Distribuição dos Depósitos')
plt.show()


# %%

# Ordenar os meses antes de exibir
ordem_meses = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
df['month'] = pd.Categorical(df['month'], categories=ordem_meses, ordered=True)

# Gráfico de barras para depósitos por profissão
plt.figure(figsize=(20, 10))
df.groupby(['job', 'deposit']).size().unstack().plot(
    kind='bar',
    stacked=False,
    ax=plt.gca(),
    color=['lightcoral', 'lightgreen'],
    edgecolor='black'
)
plt.title('Distribuição dos Depósitos por Profissão')
plt.xlabel('Profissão')
plt.ylabel('Contagem')
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
        plt.title(f'Distribuição de Depósitos por {col}')
        plt.xlabel(col)
        plt.ylabel('Contagem')
        plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# %%

i = 0
plt.figure(figsize=(20,10))
for col in df.columns:
    if df[col].dtype == 'int64':
        i += 1
        plt.subplot(4, 2, i)

        sns.histplot(df[col], bins=20, color='lightblue')
        plt.title(f'Distribuiton {col}', fontsize=12)
        plt.xlabel(col)
        plt.ylabel('Count')

plt.tight_layout()
plt.show()

# %%

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

        plt.title(f'BoxPlot de {col}')
        plt.xlabel(col)
        plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# Remover Outliers

# %%
# Função para remover outliers usando o método IQR
def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_out

# Remover outliers das colunas numéricas
numeric_columns = df.select_dtypes(include=['int64']).columns

# Remove outliers coluna a coluna (preserva mais dados)
df_clean = remove_outliers(df, numeric_columns)

# Verifica o balanceamento da variável target
print("Distribuição ANTES de remover outliers:")
print(df['deposit'].value_counts())
print("\nDistribuição DEPOIS de remover outliers:")
print(df_clean['deposit'].value_counts())

# %%

df_temp = df.copy()

df_temp['deposit'] = LabelEncoder().fit_transform(df_temp['deposit'])

# 1. Calcular a matriz de correlação
correlation_matrix = df_temp.select_dtypes(include=['number']).corr()
print(correlation_matrix)

# 2. Configurar a figura
plt.figure(figsize=(20, 10))

# 3. Criar o heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mascarar a parte superior (opcional)
sns.heatmap(
    correlation_matrix,
    #mask=mask,          # Usar máscara para ocultar triângulo superior (opcional)
    annot=True,         # Mostrar valores de correlação
    fmt=".2f",          # Formato dos números (2 casas decimais)
    cmap="coolwarm",    # Mapa de cores (ex: "viridis", "Blues", "RdBu")
    vmin=-1, vmax=1,    # Limites da escala de cores (-1 a 1)
    linewidths=0.5      # Espessura das linhas entre células
)

# 4. Personalizar o gráfico
plt.title("Matriz de Correlação entre Variáveis Numéricas", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')  # Rotacionar rótulos do eixo X
plt.tight_layout()
plt.show()

# %% [markdown]
# # 🧹 Pré-Processamento dos Dados

# %%
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

df.to_csv('Data/bank_preprocessed2.csv', index=False)

# df["month"] = df["month"].map(month_order).astype(int)
# l=['month',"contact","poutcome"]
# for i in l:
#     le=LabelEncoder()
#     df[i]=le.fit_transform(df[i].values)

# df = pd.get_dummies(df, columns = ['job','marital','education'])

# # Convertendo colunas booleanas para inteiros
# for column in df.columns:
#     if df[column].dtype == 'bool':
#         df[column] = df[column].astype(int)


# %%
print(df)

# %%
print(df.describe())

# %% [markdown]
# # ✂️ Divisão dos Dados em Treino e Teste

# %%
def create_model(hidden_size=50, learning_rate=0.001, lambda_reg=0.01, input_dim=10):
    model = Sequential([
        Dense(hidden_size, activation='relu', kernel_regularizer=l2(lambda_reg), input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Divisão treino/teste (80/20)
print('1111')
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
for train_index, test_index in split.split(df, df['deposit']):
    train, test = df.iloc[train_index], df.iloc[test_index]

# Pré-processamento
X_train, y_train = train.drop(columns=['deposit']).values, train['deposit'].values
X_test, y_test = test.drop(columns=['deposit']).values, test['deposit'].values

scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

input_dim = X_train.shape[1]
print(input_dim)
# Hiperparâmetros sem 'model__'
param_dist = {
    'hidden_size': np.arange(10, 100),
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'lambda_reg': [0.001, 0.01, 0.1, 1.0]
}

# Criar o classificador Keras
model = KerasClassifier(
    build_fn=create_model,
    epochs=100,
    verbose=0,
    input_dim=input_dim  # Passa o input_dim corretamente
)

# Busca aleatória de hiperparâmetros
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=5),
    scoring='accuracy',
    random_state=5
)

# Treinar
random_search.fit(X_train_scaled, y_train)

# Resultados
print("Melhores hiperparâmetros:", random_search.best_params_)
print("Melhor acurácia (validação):", random_search.best_score_)

best_model = random_search.best_estimator_.model
test_acc = best_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
print(f"Acurácia no teste: {test_acc:.2%}")

print(best_model)

# %%



