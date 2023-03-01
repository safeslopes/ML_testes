#!/usr/bin/env python
# coding: utf-8

# # 🧪 PREDIÇÃO DE DESLIZAMENTO USANDO SVM | TESTE

# ##  Criação do dataset

# In[3]:


# Importar bibliotecas

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[37]:


# Importar dataset

deslizamentos_dataset = pd.read_excel(r"C:\Users\Usuário\Desktop\deslizamento_teste.xlsx")
print(deslizamentos_dataset.head()) # mostra as primeiras 5 linhas


# In[38]:


# Verificar o total de linhas e colunas no dataset

deslizamentos_dataset.shape


# In[39]:


# Descrever dados estatísticos do dataset

deslizamentos_dataset.describe()


# In[40]:


# Verificar o número de ocorrências de deslizamento e não deslizamento (sim e não)

deslizamentos_dataset['deslizamento'].value_counts()


# In[41]:


X = deslizamentos_dataset.drop(columns='deslizamento', axis=1)
Y = deslizamentos_dataset['deslizamento']


# In[42]:


print (X)


# In[43]:


print(Y)


# In[44]:


scaler = StandardScaler() #criação de instância


# In[46]:


scaler.fit(X) #fiting de X


# In[47]:


standarized_data = scaler.transform(X) #transformar data


# In[48]:


X = standarized_data #alimentando os dados padronizados para a variável X


# In[49]:


print(X)
print(Y)


# In[50]:


#dividindo o conjunto de dados na proporção 80/20 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify = Y, random_state=2)


# In[51]:


print(X.shape, X_test.shape, X_train.shape)


# ## 🦾 Criação do modelo

# In[52]:


classifier = svm.SVC(kernel='linear')


# In[53]:


# Treinar modelo usando o dataset de teste
classifier.fit(X_train, Y_train)


# In[58]:


# Acurácia nos dados de treinamento
train_pred = classifier.predict(X_train)
accuracy_train = accuracy_score(train_pred, Y_train)


# In[59]:


print("Pontuação de acurácia dos dados de treinamento = {}".format(accuracy_train))


# In[64]:


#Acurácia no dados de teste
test_pred = classifier.predict(X_test)
accuracy_test = accuracy_score(test_pred, Y_test)


# In[65]:


print("Pontuação de acurácia dos dados de teste = {}".format(accuracy_test))


# ## 🔮 Sistema de predição

# In[66]:


input_data = (200, 80, 24, 1)

# Transformar dados de entrada em um array
data_changed = np.asarray(input_data)

# Remodelar para predição de apenas uma instância
data_reshaped = data_changed.reshape(1,-1)

# Padronizar dados de entrada
std_data = scaler.transform(data_reshaped)
print(std_data)

# Predição com os valores de entrada
prediction = classifier.predict(std_data)
if prediction == 1:
    print ("Vai ocorrer deslizamento")
else:
    print("Não vai ocorrer deslizamento")


# In[68]:


input_data = (10, 15, 24, 1)

# Transformar dados de entrada em um array
data_changed = np.asarray(input_data)

# Remodelar para predição de apenas uma instância
data_reshaped = data_changed.reshape(1,-1)

# Padronizar dados de entrada
std_data = scaler.transform(data_reshaped)
print(std_data)

# Predição com os valores de entrada
prediction = classifier.predict(std_data)
if prediction == 1:
    print ("Vai ocorrer deslizamento")
else:
    print("Não vai ocorrer deslizamento")


# In[69]:


input_data = (0, 86, 25, 0)

# Transformar dados de entrada em um array
data_changed = np.asarray(input_data)

# Remodelar para predição de apenas uma instância
data_reshaped = data_changed.reshape(1,-1)

# Padronizar dados de entrada
std_data = scaler.transform(data_reshaped)
print(std_data)

# Predição com os valores de entrada
prediction = classifier.predict(std_data)
if prediction == 1:
    print ("Vai ocorrer deslizamento")
else:
    print("Não vai ocorrer deslizamento")


# ## 📊 Outras métricas do modelo

# In[79]:


# Importar bibliotecas
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# In[101]:


#Matriz de confusão

test_pred = classifier.predict(X_test)
accuracy_test = accuracy_score(test_pred, Y_test)

cm = confusion_matrix(Y_test, test_pred)
plot_confusion_matrix(classifier, X_test, Y_test, cmap=plt.cm.Reds)
plt.show()


# In[103]:


# Curva de aprendizado 

'''
O gráfico mostra como a precisão do modelo melhora à medida que o número de exemplos de treinamento 
aumenta e também mostra se o modelo está sofrendo  overfitting ou underfitting.
'''

train_sizes, train_scores, test_scores = learning_curve(
    svm.SVC(kernel='linear'),
    X_train, Y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title('Curva de Aprendizado')
plt.xlabel('Exemplos de treinamento')
plt.ylabel('Score')

plt.plot(train_sizes, train_scores_mean, 'o-', color='red', label='Score de Treinamento')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1)

plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Cross-Validation Score')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)

plt.legend(loc='best')
plt.show()


# In[ ]:




