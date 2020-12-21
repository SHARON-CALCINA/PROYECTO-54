# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 02:10:27 2020

@author: SHARON
"""

# DETERMINAR QUE HACE QUE CIERTO TIPO DE BARRA DE CHOCOLATE SEA LA MEJOR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("chocolate.csv")
#eliminamos columnas con datos que no necesitamos
data = data.drop(['ref','company_location','review_date','specific_bean_origin_or_bar_name'], axis = 1) 
data = data.drop(['Unnamed: 0'], axis = 1)
#print(data)
#eliminamos columnas con un solo valor o valor único
nunique = data.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index

data=data.drop(cols_to_drop, axis=1)

#print(data)
#EXPLORACION DEL CONJUNTO DE DATOS
#Comprensión de datos
data_comprenhension = {'ColumnName' : ['Company', 'country_of_bean_origin', 'cocoa_percent','rating','counts_of_ingredients',
'cocoa_butter','vanilla','lecithin','salt','sugar','sweetener_without_sugar','first_taste','second_taste',
'third_taste','fourth_taste'], 
'DataType': ['String', 'String', 'Integer','Float', 'Integer', 
            'bool', 'bool', 'bool','bool', 'bool', 'bool', 
            'String', 'String', 'String', 'String'],
'Values' : ['A. Morin','France','70','4','4','have_cocoa_butter','have_not_vanila','have_lecithin','have_not_salt',
           'have_sugar','have_not_sweetener_without_sugar','oily','nut','caramel','raspberry'],
'description': ['name of company wich makes chocolate bar',
                'origin country where chocolate bean is cultivated',
                'how much percentage of cocoa is used in the creation of chococlate bar',
                'rating of chocolate bar between 1-5, considering that 1.0 - 1.9 = Unpleasant (desagradable), 2.0 - 2.9 = Disappointing(Decepcionante), 3.0 - 3.49 = Recommended (recomendado), 3.5 - 3.9 = Highly Recommended (altamente recomendado), 4.0 - 5.0 = Outstanding (excepcional)',
                'quantity of ingredients that take part in making chocolate bar',
                'Is chocolate butter one of the ingredients o chocolate bar?',
                'Is vanilla one of the ingredients o chocolate bar?',
                'Is lecithin one of the ingredients o chocolate bar?',
                'Is salt one of the ingredients o chocolate bar?',
                'Is sugar one of the ingredients o chocolate bar?',
                'Is sweetener_without_sugar one of the ingredients o chocolate bar?',
                'wich is one of the flavors of the chocolate bar - fisrt',
                'wich is one of the flavors of the chocolate bar - second',
                'wich is one of the flavors of the chocolate bar - third',
                'wich is one of the flavors of the chocolate bar - fourth']
}
data_frame = pd.DataFrame(data_comprenhension)
#print (data_frame)

# Decidir sobre columna objetivo
#MATRIZ VAR INDEP.
x1= data.iloc[:, :3].values
x2= data.iloc[:, 4:].values
x=x1
x=np.append(x1,x2,axis=1)
#MATRIZ VAR DEPENDANT
y= data.iloc[:, 3].values
#y= pd.DataFrame(y)
#y.to_csv("y", index= False)
x= pd.DataFrame(x)
#x.to_csv("IndVar", index= False)

#visualizacion de los datos
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.distplot(data['rating'], bins=10)
#plt.show()

# NO es una ditribución normal
# ya que queremos determinar como el q una barra de chococlate
#recibe mayor rating,Vamos a RECORTAR los datos en la zona donde
#el rating sea menor: entre 1 y 2.5
#data = data[(data['rating'] > 2.5)]
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.distplot(data['rating'], bins=5)
#plt.show()
#----------------MANEJAR VALORES FALTANTES
x=x.fillna('have_no_taste')

#x.to_csv("fill empty", index= False)
#---------------MANEJO DE DATOS CATEGORICOS
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 
x.columns=['company','country_of_bean_origin','cocoa_percent',
          'counts_of_ingredients','cocoa_butter','vanilla','lecithin',
          'salt','sugar','sweetener_without_sugar','first_taste',
          'second_taste','third_taste','fourth_taste']
  
x['company']= le.fit_transform(x['company']) 
x['country_of_bean_origin']= le.fit_transform(x['country_of_bean_origin'])
x['cocoa_butter']= le.fit_transform(x['cocoa_butter']) 

x=x.replace({'have_cocoa_butter':1, 'have_not_cocoa_butter':0,
           'have_vanila':1,'have_not_vanila':0,
           'have_lecithin':1,'have_not_lecithin':0,
           'have_salt':1,'have_not_salt':0,
           'have_sugar':1,'have_not_sugar':0,
           'have_sweetener_without_sugar':1,'have_not_sweetener_without_sugar':0,
           })
#x['vanilla']= le.fit_transform(x['vanilla'])
#x['lecithin']= le.fit_transform(x['lecithin']) 
#x['salt']= le.fit_transform(x['salt'])
#x['sugar']= le.fit_transform(x['sugar']) 
#x['sweetener_without_sugar']= le.fit_transform(x['sweetener_without_sugar'])
x['first_taste']= le.fit_transform(x['first_taste']) 
x['second_taste']= le.fit_transform(x['second_taste'])
x['third_taste']= le.fit_transform(x['third_taste']) 
x['fourth_taste']= le.fit_transform(x['fourth_taste'])
#x.to_csv("replaceVal", index= False)
x= pd.DataFrame(x)
y= pd.DataFrame(y)
y.columns[0]
#redefining rating values to 0,1,2,3,4
y=y.replace({1:0, 1.5:0, 1.75:0, 2:0, 2.25:1, 2.5:1, 2.6:1, 2.75:1, 3:2, 3.25:2, 3.5:3, 3.75:3, 4:4})
#print(y)

#x.hist()
#plt.show()
# As most of data refered to chocolate bar ingredients:
    # has sugar (azucar)
    # doesn't have sweetener whitout sugar (edulcorante sin azucar)
    # doesn't have salt
# we proceed to get rid of those columns -> redundat data
x = x.drop(['salt', 'sugar','sweetener_without_sugar'], axis = 1)
#x.hist()
#plt.show()
# cannot do the same with flavors 'cause data in each one vary and were filled up in no order way

x= pd.DataFrame(x)
y= pd.DataFrame(y)
data2 = np.append(x,y, axis=1)
data2=pd.DataFrame(data2)
data2.columns=['company','country_of_bean_origin','cocoa_percent',
          'counts_of_ingredients','cocoa_butter','vanilla','lecithin',
          'first_taste','second_taste','third_taste','fourth_taste','rating']

#A continuación, creamos una matriz de correlación que mide las relaciones lineales entre las variables.
#corrMatrix = data2.corr()
#corrMatrix.style.background_gradient(cmap='coolwarm')
#sns.heatmap(corrMatrix, annot=True)
#plt.show()
#print(corrMatrix)

# Observations:
#To fit a linear regression model, we select those features which have a high correlation with our target 
#variable Rating. By looking at the correlation matrix we can see that country_of_bean_origin is the only
#positive correlation with rating (0.029) where as vanilla, counts_of_ingredients and cocoa_percent have a
# high negative correlation with rating compared with thw others(-0.14. -0.072. -0.073).
#An important point in selecting features for a linear regression model is to check for multi-co-linearity. 
#The features cocoa_butter, counts_of_ingredients have a correlation of -0.75. These feature pairs are 
#strongly negative correlated to each other. We should not select both these features together for training
# the model. Same goes for the features vanilla lecithin, counts_of_ingredient and vanilla which have a 
#correlation of 0.74 and 0.69.

#Therefore, to apply the model we choose the features: all flavors, country_of_bean_origin, vanilla (between vanilla and 
#counts_of_ingredients we go for the first one, because vanilla has more correlation than the other three 
#ingredients and so, the second is kind of much more general including all ingredients, some of wich we got 
#rid off, so it wouldn't make so much sense to go for that one.

data2 = data2.drop(['company','cocoa_percent',
          'counts_of_ingredients','cocoa_butter','lecithin'], axis = 1)
x= pd.DataFrame(x)

x = x.drop(['company','cocoa_percent','counts_of_ingredients','cocoa_butter','lecithin'], axis=1)

#print(x)          
#x.columns=['country_of_bean_origin','vanilla','first_taste','second_taste','third_taste','fourth_taste']

x=x.values.tolist()
y=y.values.tolist()
#x=np.array(x)
#y=np.array(y)
#print(x)
#print(y)

from sklearn import tree

classifier= tree.DecisionTreeClassifier()
classifier.fit(x,y)
res=classifier.predict([[37, 1, 42, 22, 127, 38 ]])
print("'**************resultado prediccion  1111************'")
print(res)
tree = tree.DecisionTreeClassifier(random_state=0).fit(x, y)
data2= pd.DataFrame(data2)
data2.columns=['country_of_bean_origin','vanilla','first_taste','second_taste','third_taste','fourth_taste', 'rating']

#sns.pairplot(data2, hue = 'rating', vars = ['company','country_of_bean_origin','cocoa_percent',
 #         'counts_of_ingredients','cocoa_butter','vanilla','lecithin',
  #        'first_taste', 'second_taste','third_taste','fourth_taste'])
feature_names = ['country_of_bean_origin','vanilla','first_taste','second_taste','third_taste','fourth_taste']
#print(data2)
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
dot_data = export_graphviz(tree,
                           feature_names=feature_names)
graph = graph_from_dot_data(dot_data)
#graph.write_png('tree.png')


scoresTreeTest=[]
scoresNeuronTest=[]
scoresTreeTrain=[]
scoresNeuronTrain=[]
scoresGauss=[]
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
iter_for_prediction = cv.split(x, y)
#for train_index, test_index in cv.split(x,y):
    #print("TRAIN:", train_index, "TEST:", test_index)


for index, (train_index, test_index) in enumerate(iter_for_prediction):
    print('***************   CON ARBOLES DE DECISION *************')
    
    print(index)
    estimador = classifier.fit(x,y)
    score = cross_val_score(classifier, x, y, scoring='accuracy', cv=[(train_index, test_index)])
    print('Estimador',estimador)
    print('Score',score)
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    for i in range (0, len(train_index)):
        x_train.append(x[train_index[i]])
    for i in range (0, len(test_index)):
        x_test.append(x[test_index[i]])
    for i in range (0, len(train_index)):
        y_train.append(y[train_index[i]])
    for i in range (0, len(test_index)):
        y_test.append(y[test_index[i]])
    classifier.fit(x_train,y_train)
    res=classifier.predict([[37, 1, 42, 22, 127, 38 ]])
    predictions=classifier.predict(x_test)
    print("-------resultado prediccion --------")
    print(res)
    yhat_test = classifier.predict(x_test)
    acc = accuracy_score(y_test, yhat_test)
    #presicion
    print('acuuracy: ', acc)
    scoresTreeTest.append(classifier.score(x_test,y_test))
    scoresTreeTrain.append(classifier.score(x_train,y_train))
    print('Score Training', classifier.score(x_train,y_train))
    print('Score Test', classifier.score(x_test,y_test))
    from sklearn import tree
    depth = []
    for i in range(3,20):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf = classifier.fit(x_train,y_train)
        depth.append((i,clf.score(x_test,y_test)))
    print (depth)
    from sklearn.metrics import classification_report
    print('REPORTE DEL CLASIFICADOR ')
    print(classification_report(y_test, predictions))
    from sklearn.metrics import confusion_matrix
    print('MATRIZ DE CONFUSION ')
    print(confusion_matrix(y_test, predictions))
    
    print('****************** CON REDES NEURONALES ****************')
    from sklearn.neural_network import MLPClassifier
    mlp=MLPClassifier(hidden_layer_sizes=[5,4], max_iter=500, alpha=0.0001, 
                                 activation= 'relu', solver='adam', random_state=21, tol=0.0001)

    mlp.fit(x_train, y_train)
    predictions2=mlp.predict(x_test)
    scoresNeuronTest.append(mlp.score(x_test, y_test))
    scoresNeuronTrain.append(mlp.score(x_train, y_train))
    print('score mlp Test:', mlp.score(x_test, y_test))
    print('score mlp Training:', mlp.score(x_train, y_train))
    from sklearn.metrics import classification_report
    print('REPORTE DEL CLASIFICADOR ')
    print(classification_report(y_test, predictions2))
    from sklearn.metrics import confusion_matrix
    print('MATRIZ DE CONFUSION ')
    print(confusion_matrix(y_test, predictions2))
    
    
    from sklearn.metrics import accuracy_score
    
    predictions_train = mlp.predict(x_train)
    print('acuuracy mlp traim: ', accuracy_score(y_train, predictions_train))
    predictions_test = mlp.predict(x_test)
    print('acuuracy mlp test: ',accuracy_score(y_test, predictions_test))

    
    print('****************** GAUSS NAIVE BAYES ****************')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    randomf = RandomForestClassifier(random_state=0)
    gnb = GaussianNB()
    sk_clasificador = {'GaussianNB':gnb,
                       'RandomForest':randomf}
    for clasifier in sk_clasificador:
        estimador = sk_clasificador[clasifier]
        scoreGauss = cross_val_score(estimador, x, y, scoring='accuracy', cv=[(train_index, test_index)])
        print('score Gauss:',estimador, scoreGauss)
        scoresGauss.append(scoreGauss)
    gnb.fit(np.array(x_train), np.array(y_train))
    predictionsgnb=gnb.predict(x_test)
    print('REPORTE DEL CLASIFICADOR ')
    print(classification_report(y_test, predictionsgnb))
    print('MATRIZ DE CONFUSION ')
    print(confusion_matrix(y_test, predictionsgnb))
    
   

print('**********  SCORES  *************')
print('tree scores')
print(scoresTreeTest)
print(scoresTreeTrain)
print('mlp scores')
print(scoresNeuronTest)
print(scoresNeuronTrain)
print('gauss scores')
print(scoresGauss)
