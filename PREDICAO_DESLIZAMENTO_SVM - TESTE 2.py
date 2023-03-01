#!/usr/bin/env python
# coding: utf-8

# # 🧪 PREDIÇÃO DE DESLIZAMENTO USANDO SVM | TESTE II

# ## Criação do dataset

# In[30]:


# Importar bibliotecas

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[5]:


# Importar dataset

deslizamentos_dataset = pd.read_excel(r"C:\Users\e109513\OneDrive - Tokio Marine Seguradora S A\Área de Trabalho\deslizamentos_testes_2.1.xlsx")
print(deslizamentos_dataset.head()) # mostra as primeiras 5 linhas


# In[6]:


# Verificar o total de linhas e colunas no dataset

deslizamentos_dataset.shape


# In[7]:


# Descrever dados estatísticos do dataset

deslizamentos_dataset.describe()


# In[8]:


# Verificar o número de ocorrências de deslizamento e não deslizamento (sim e não)

deslizamentos_dataset['deslizamento'].value_counts()


# In[9]:


X = deslizamentos_dataset.drop(columns='deslizamento', axis=1)
Y = deslizamentos_dataset['deslizamento']


# In[10]:


print (X)


# In[11]:


print(Y)


# In[12]:


scaler = StandardScaler() #criação de instância


# In[13]:


scaler.fit(X) #fiting de X


# In[14]:


standarized_data = scaler.transform(X) #transformar data


# In[15]:


X = standarized_data #alimentando os dados padronizados para a variável X


# In[16]:


print(X)
print(Y)


# In[17]:


#dividindo o conjunto de dados na proporção 80/20 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify = Y, random_state=2)


# In[18]:


print(X.shape, X_test.shape, X_train.shape)


# ## 🦾 Criação do modelo
# 

# In[19]:


classifier = svm.SVC(kernel='linear')


# In[20]:


# Treinar modelo usando o dataset de teste
classifier.fit(X_train, Y_train)


# In[21]:


# Acurácia nos dados de treinamento
train_pred = classifier.predict(X_train)
accuracy_train = accuracy_score(train_pred, Y_train)


# In[22]:


print("Pontuação de acurácia dos dados de treinamento = {}".format(accuracy_train))


# In[23]:


#Acurácia no dados de teste
test_pred = classifier.predict(X_test)
accuracy_test = accuracy_score(test_pred, Y_test)


# In[24]:


print("Pontuação de acurácia dos dados de teste = {}".format(accuracy_test))


# ## 🔮 Sistema de predição 

# In[25]:


input_data = (600, 99, 24)

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

# In[38]:


from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt


# In[39]:


cm = confusion_matrix(Y_test, test_pred)

# Plot confusion matrix
plot_confusion_matrix(classifier, X_test, Y_test, cmap=plt.cm.Reds)
plt.show()


# In[40]:


from sklearn.model_selection import learning_curve


# In[41]:


train_sizes, train_scores, test_scores = learning_curve(classifier, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.show()


# In[42]:


from sklearn.metrics import roc_curve, auc


# In[43]:


#Acurácia no dados de teste
test_pred = classifier.predict(X_test)
accuracy_test = accuracy_score(test_pred, Y_test)

# ROC curve
fpr, tpr, _ = roc_curve(Y_test, classifier.decision_function(X_test))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[44]:


from sklearn.metrics import precision_recall_curve


# In[45]:


# Predict probabilities of positive class for test set
y_score = classifier.decision_function(X_test)

# Compute precision and recall values for different thresholds
precision, recall, thresholds = precision_recall_curve(Y_test, y_score)

# Plot the precision-recall curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[50]:


# Obter os vetores de suporte a partir do treinamento
support_vectors = classifier.support_vectors_

# Plotar os pontos do dataset de treinamento
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)

# Plotar os vetores de suporte em amarelo 
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='r')

plt.show()


# In[51]:


from mpl_toolkits.mplot3d import Axes3D


# In[57]:


# Create a mesh grid of points to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1),
                         np.arange(z_min, z_max, 0.1))

# Make predictions for each point in the mesh grid
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Create a 3D plot of the decision boundary
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
ax.set_xlabel('Indice Pluviometrico')
ax.set_ylabel('Umidade')
ax.set_zlabel('Temperatura')
ax.set_title('Limite de decisão')
ax.view_init(30, 45)
ax.plot_surface(xx, yy, Z, alpha=0.2)
plt.show()

