import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('AA_TP1/Data/bank.csv')

print(df.head())

data = df.values
X = data[:,0:-1]
y = data[:,-1]

print(X.shape)
print(y.shape)
y = y.reshape(y.shape[0],1)
print(y.shape)

print(df.isnull().sum())
print(df.describe())
print(df['deposit'].value_counts())

# Ver pie chart com a percentagem de depositos
plt.pie(df['deposit'].value_counts(),labels=['no', 'yes'],autopct='%1.1f%%',  startangle=-90, colors=['red','green'])
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
    color=['red','green'],
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
            color=['red','green'],
            edgecolor='black')
        plt.title(f'Distribution of deposits by {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

        

