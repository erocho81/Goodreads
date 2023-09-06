#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Importacion de los datasets
#Dataset import

goodreads = pd.read_csv ('D:/Análisis de datos/DATASETS/Goodreads/books.csv', on_bad_lines='skip')


# In[3]:


#Revisamos uno de los datasets
#We check one of the datasets
goodreads.head()


# In[4]:


#Let's check if there are Na in some of the columns
#Queremos revisar si hay NA's en alguna de las columnas que utilizaremos
goodreads.average_rating.isna().any()


# In[5]:


goodreads.title.isna().any()


# In[6]:


#Revisamos las columnas de uno de los datasets
#We check the columns for one of the datasets
goodreads.columns


# In[7]:


#Removing some columns
goodreads = goodreads.drop(columns=['isbn', 'isbn13'])


# In[8]:


goodreads.columns


# In[9]:


#Creation of new column "Year"
goodreads['Year'] = goodreads['publication_date'].str.strip().str[-4:]


# In[10]:


#Cheking data types
goodreads.dtypes


# In[11]:


#rename incorrect column

goodreads = goodreads.rename(columns={'  num_pages': 'num_pages'})


# In[12]:


goodreads.dtypes


# In[12]:


#Books with most ratings_count

df1 = goodreads.sort_values('ratings_count',ascending = False).groupby('title').head(5)
print (df1)


# In[13]:


#Let's check the empty values:

#Revisamos valores vacíos:

total = goodreads.isnull().sum().sort_values(ascending=False)
percent = (goodreads.isnull().sum()/goodreads.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data['Total'] > 0]


# In[14]:


#Tenemos que volver a cambiar los 0 de Year por NAN porque sino nos contará los años 0, 
#ya que si creamos visualizaciones aparecerían valores incorrectos contados como 0, afectando a los gráficos:

#However for the following visualizations we will change back the Year 0's to Na's, 
#to avoid a count of the 0 values that would mess the graphs:

goodreads.publication_date.replace(0, np.nan, inplace=True)


# In[15]:


#Ahora realizamos las visualizaciones para los valores numéricos del df.
#Here we show the visualizations for the numerical values of the df.

goodreads.hist(bins=50, figsize=(30,20))


# In[15]:


goodreads.hist(bins=50, figsize=(30,20), column=["average_rating", "num_pages", "ratings_count"])


# In[16]:


#Vamos a visualizar los valores medios de "average_rating" y "num_pages" por año.

#Let's plot the average "average_rating" y "num_pages" per year.

goodreads.groupby('Year')["num_pages"].mean().plot()


# In[17]:


goodreads.groupby('Year')["average_rating"].mean().plot()


# In[41]:


#Podemos revisar las medias, min, max, count de algunas de las columnas numéricas.
#We can also check avg, min, max, std, count for some of the numerical values of the df.

goodreads[['average_rating','num_pages','ratings_count','text_reviews_count']].describe()


# In[19]:


#This is a bivariant visualization for average_rating and ratings_count.

var = 'average_rating'
data = pd.concat([goodreads['ratings_count'], goodreads['average_rating']], axis=1)
data.plot.scatter(x='average_rating', y='ratings_count')


# In[20]:


#Cantidad de libros según lenguaje
goodreads["language_code"].value_counts().plot(kind='bar')


# In[21]:


#Realizamos un análisis multivariante para las variables numéricas. 
#Vemos que hay una relación clara entre el ratings_count y text_reviews_count.

#We create a multivariant visualization.
#We can find a high correlation between ratings_count and text_reviews_count.

#Multivariante sin normalizar:

corrmat = goodreads.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)


# In[22]:


#Bivariante entre language_code y average_rating con boxplot. 

#The following is a bivariant boxplot visualization between language_code and average_rating. 


var = 'language_code'
data = pd.concat([goodreads['average_rating'], goodreads[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="average_rating", data=data)
plt.xticks(rotation=90);


# In[23]:


#Drop de filas con na, para evitar problemas a la hora de normalizar y aplicar modelos:

goodreads_nona = goodreads.dropna()


# In[24]:


#Using mean instead of dropping na's would get worse results for a linear regression:

#sales_rtg_nona = sales_rtg.fillna(sales_rtg.mean())

goodreads_nona.head()


# In[25]:


#LINEAR REGRESSION

#Entrenamiento y Test.
#let's study the relation between average_rating and ratings_count

from sklearn.model_selection import train_test_split

X = goodreads_nona.average_rating.values #This is the average_rating column

Y = goodreads_nona.ratings_count.values #This is the ratings_count column


X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0) 


# In[26]:


#We use reshape to avoid the following error for lear regression: Expected 2D array, got 1D array instead
X_train= X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[28]:


#Fitting Simple Linear Regression Model to the training set
#Removed normalize True due to error, not finding normalize

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


# In[29]:


#The coefficient of determination, denoted as 𝑅², tells you which amount of variation in 𝑦 can be explained
#by the dependence on 𝐱, using the particular regression model. 
#A larger 𝑅² indicates a better fit and means that the model can better explain the variation
#of the output with different inputs.

#The value 𝑅² = 1 corresponds to SSR = 0. That’s the perfect fit, since the values
#of predicted and actual responses fit completely to each other.

#In this case the value is only 0.0012 which is low for the model

r_sq = regressor.score(X_train, Y_train)
print(f"coefficient of determination: {r_sq}")


# In[30]:


print(f"intercept: {regressor.intercept_}")


# In[31]:


print(f"slope: {regressor.coef_}")


# In[ ]:


#The value of 𝑏₀ is approx -27569
#This illustrates that the model predicts the response -27569 when X is zero.
#The value 𝑏₁ = 11523.62 means that the predicted response rises by 11523.62 when X is increased by one.


# In[33]:


#Predicted Response

predicted = regressor.predict(X_test)

print(f"predicted response:\n{predicted}")


# In[34]:


#Visualization of training results

plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')


# In[35]:


#Visualization of the test results

plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')


# In[36]:


#We can use this fitted model to calculate the outputs based on new inputs:

x_new = np.arange(5).reshape((-1, 1))
x_new


# In[47]:


#LOGISTIC REGRESSION
#Vamos a crear un subset categorico.

#Let's create a categorical subset.

goodreads_categorical = goodreads_nona[['language_code']]
goodreads_categorical 


# In[48]:


#Creamos nuevas columnas categóricas binarizadas:

#We create the new binary columns:

import pandas as pd

goodreads_categorical = pd.concat([pd.get_dummies(goodreads_categorical[col], prefix=col) for col in goodreads_categorical], axis=1)


# In[49]:


#We concat some of the numeric columns to the new categorical ones.

#Unimos varias variables numéricas con las columnas categóricas nuevas.

df_categ = pd.concat([goodreads_nona[['average_rating', 'num_pages','ratings_count']], goodreads_categorical], axis=1)
df_categ.head()


# In[50]:


goodreads_categorical.head()


# In[51]:


df_categ.columns


# In[52]:


#LOGISTIC REGRESSION

#Vamos a crear un nuevo par de test/train
#We are going to create a new test/train pair

from sklearn.model_selection import train_test_split

X2 = df_categ.drop('language_code_spa', 1)
y2 = df_categ.language_code_spa

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.30, random_state=42)


# In[60]:


y_test_2


# In[53]:


#Aplicamos el modelo
#We apply the model

from sklearn import linear_model, datasets

logreg = linear_model.LogisticRegression(max_iter=600, solver='lbfgs')
model = logreg.fit(X_train_2, y_train_2)
model


# In[54]:


#We check the predicted data
#Revisamos la información predicha

predicted_2 = model.predict(X_test_2)
predicted_2


# In[55]:


#Creamos una matriz de confusión para revisar cuantos datos han sido correctamente clasificados.
#We create a confusion matrix to check how many data have been correctly/incorrectly classified.



import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

#predicted_2 = np.round(predicted_2)
matrix2 = confusion_matrix(y_test_2, predicted_2)
sns.heatmap(matrix2, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicción")
plt.ylabel("real")
plt

#We see that 3283 are True positives, the model predicted true, and it's true.
#There are also 54 false negatives. The model predicted them as false but they are true (to recheck)


# In[56]:


#Esta es la accuracy que obtenemos del modelo.
#This is the accuracy obtained:

from sklearn.metrics import accuracy_score

accuracy_score(y_test_2, predicted_2)


# In[57]:


#We can check more detailed information about the trained model:

from sklearn.metrics import classification_report

report = classification_report(y_test_2, predicted_2)
print(report)

#recall is the sensitivity, which measures how food is the model at predicting positives.
#f1-score is the harmonic mean of precision and sensitivity, it considers both false positive and false negatives,
# and is good for imbalanced datasets


# In[58]:


# DECISION TREES

# Load libraries
# Cargamos las librerías
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[68]:


#Decision Trees can't work with decimal numbers (continuous), so we need to change the average_rating column:

#We *100 the column, to create a new column
df_categ['average_rating_100'] = df_categ.average_rating * 100

#Convert the new column to integer
df_categ['average_rating_100'] = df_categ['average_rating_100'].astype('int')

df_categ.head()


# In[69]:


#Split dataset in features and target variable. Meta_score will be the variable that we want to predict.

#Dividimos de nuevo el dataset en features y target. Meta_score será la variable que queremos predecir

feature_cols = ['num_pages', 'ratings_count','language_code_ale','language_code_en-CA','language_code_en-GB','language_code_en-US','language_code_eng','language_code_fre','language_code_ger','language_code_spa']
X3 = df_categ[feature_cols] # Features
y3 = df_categ.average_rating_100 # Target variable


# In[60]:


X3


# In[70]:


# Split dataset into training set and test set
#Creamos de nuevo test/entrenamiento

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X3, y3, test_size=0.3, random_state=1) # 70% training and 30% test


# In[71]:


# Creation of Decision Tree classifer object
# Creamos el objeto con el clasificador del arbol.

clf = DecisionTreeClassifier()


# In[72]:


# Train Decision Tree Classifer
# Entrenamos el arbol
clf = clf.fit(X_train_3,y_train_3)


# In[73]:


#Predict the response for test dataset
#Predecimos la respuesta para el dataset test

y_pred_3 = clf.predict(X_test_3)


# In[74]:


# Model Accuracy, 1,85%
# La precisión del modelo es del 1,85%

print("Accuracy:",metrics.accuracy_score(y_test_3, y_pred_3))


# In[75]:


#KNN Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Se asignan 2 variables a X y una a y:
#We assign 2 variables to X and one to y:
X4 = df_categ[['num_pages','ratings_count']].values
y4 = df_categ['average_rating_100'].values


# In[76]:


#realizamos de nuevo la división test/training
#we split the data into test/training again:

X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X4, y4, random_state=0)
scaler = MinMaxScaler()
X_train_4 = scaler.fit_transform(X_train_4)
X_test_4 = scaler.transform(X_test_4)


# In[82]:


#Definimos k como 3 ya que da un poco mejor accuracy
#We have assigned k=3 since it seems to provide a slightly better accuracy:

n_neighbors = 3

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train_4, y_train_4)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_4, y_train_4)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_4, y_test_4)))


# In[83]:


#Precisión del modelo:

pred = knn.predict(X_test_4)
print(confusion_matrix(y_test_4, pred))
print(classification_report(y_test_4, pred))


# In[84]:


#Hacemos fit knn de X4, y4, no de test/training
#We fit X4, y4, not training/test

clf2= knn.fit(X4, y4)


# In[86]:


#Con esto podemos obtener una predicción. Para 300 pgs y 2000 review_count obtenemos un average score de 359 (3,59).

#With this we can try to make a prediction.For 300 pages and 2000 review_count we get an average score de 359 (3,59).

print(clf2.predict([[300, 2000]]))


# In[87]:


#The average score increases for 400 pages and 5000 review_count

print(clf2.predict([[400, 5000]]))


# In[88]:


#The average score decreases if we increase the review_count

print(clf2.predict([[400, 10000]]))

