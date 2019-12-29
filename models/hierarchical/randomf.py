###################################################################################
# Clasificador: randomf
#
# Autor: Carlos Muñiz Solaz
# Fecha: Enero 2020
#
# Utiliza clasificador random forest para clasificar el data set de validación de NB 15
###################################################################################

import numpy as np
import random
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

import seaborn as sn

from sklearn.preprocessing import StandardScaler
from time import time

import matplotlib
matplotlib.use('Qt5Agg')

np.random.seed(2019)  # Para garantizar la reproducibilidad
random.seed(2019)


# Funcion que permite contar el numero de etiquetas
def proporcion_etiquetas(y):
    _, count = np.unique(y, return_counts=True)
    return np.round(np.true_divide(count, y.shape[0]) * 100, 2)




###################################################################################################
# Obtenemos el nombre de las caracteristicas
###################################################################################################
data_df = pd.read_csv('/home/carlos/Projects/tfm/nb15/working//training_subset_rand.csv')
features = list ( data_df.columns.values[0:19] ) # Nombre de las columnas

y = data_df.iloc[:,19]
labels = y  # Clasificación de los ataques

###################################################################################################
# Leemos los ficheros
###################################################################################################


#data_train = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/60percent.csv",delimiter=",", skiprows=1)
data_train = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_total_numeric_testing_WR_19FS-60percent.csv",delimiter=",", skiprows=1)
X_train = data_train[:,0:19]
y_train = data_train[:,-1]

#data_test = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/40percent.csv",delimiter=",", skiprows=1)
data_test = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_total_numeric_testing_WR_19FS-40percent.csv",delimiter=",", skiprows=1)
X_test = data_test[:,0:19]
y_test = data_test[:,-1]

# 'Normal','Backdoor','Analysis','Fuzzers','Shellcode','Reconnaissance','Exploits','DoS','Worms','Generic'

print("Número de registros para entrenar: {}".format(X_train.shape[0]))
print("Número de registros para test: {}".format(X_test.shape[0]))

print("\nProporción de las etiquetas en el conjunto original: {}".format(proporcion_etiquetas(labels)))
print("Proporción de las etiquetas en el conjunto de entrenamiento: {}".format(proporcion_etiquetas(y_train)))
print("Proporción de las etiquetas en el conjunto de test: {}".format(proporcion_etiquetas(y_test)))

###################################################################################################
# Fase de entrenamiento
###################################################################################################

start = time()

# Normalizamos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)



rdf = RandomForestClassifier(n_estimators=15, oob_score=True, random_state=2019)


# Entrenamos el modelo
rdf.fit(X_train, y_train)

# Precisión usando validación cruzada del conjunto de entrenamiento
scores = cross_val_score(rdf, X_train, y_train, cv = 4)

# Precisión out-of-bag
print ('Out-of-bag Accuracy : {}'.format(rdf.oob_score_))

# Out of the bag error
# oob_error = 1 - clf.oob_score_

print("Training %.2f seconds:" % ((time() - start)))
print ('Cross-validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))

###################################################################################################
# Fase de testing
###################################################################################################
start = time()

# Normalizamos
X_test = sc.transform(X_test)




rdf.predict(X_test)
y_pred = rdf.predict(X_test)
target_names = ['Normal','Backdoor','Analysis','Fuzzers','Shellcode','Reconnaissance','Exploits','DoS','Worms','Generic']

acc = accuracy_score( y_test, y_pred)
conf=confusion_matrix (y_test, y_pred)
print ()
print(classification_report(y_test, y_pred, target_names=target_names))
print ()
print (conf)
print ()
print ("Normal:     "+str(float(conf[0,0])/np.count_nonzero(y_test==0)))
print ("Backdoor:   "+str(float(conf[1,1])/np.count_nonzero(y_test==1)))
print ("Analysis:   "+str(float(conf[2,2])/np.count_nonzero(y_test==2)))
print ("Fuzzers:    "+str(float(conf[3,3])/np.count_nonzero(y_test==3)))
print ("Shellcode:  "+str(float(conf[4,4])/np.count_nonzero(y_test==4)))
print ("Recon:      "+str(float(conf[5,5])/np.count_nonzero(y_test==5)))
print ("Exploits:   "+str(float(conf[6,6])/np.count_nonzero(y_test==6)))
print ("DoS:        "+str(float(conf[7,7])/np.count_nonzero(y_test==7)))
print ("Worms:      "+str(float(conf[8,8])/np.count_nonzero(y_test==8)))
print ("Generic:    "+str(float(conf[9,9])/np.count_nonzero(y_test==9)))
print("Testing %.2f seconds:" % ((time() - start)))
print ('Test Accuracy : {}'.format(accuracy_score(y_test, y_pred)))
print ()


#####################################################################################################
# Visualizamos la matriz de confusión
#####################################################################################################
n_classes = 10
df_cm = pd.DataFrame(conf, range(n_classes), range(n_classes))
plt.figure(figsize = (20,14))

attacks_labels = ['Normal','Backdoor','Analysis','Fuzzers','Shellcode','Reconnaissance', 'Exploits', 'DoS','Worms','Generic']

sn.set(font_scale=1.4)
sn.heatmap(df_cm,  annot=True, annot_kws={"size": 10}, fmt='g', xticklabels=attacks_labels, yticklabels=attacks_labels)
sn.set(font_scale=1) # font size 2
plt.show()


#############################################################################
# Todas las importancia de cada caracteristicas. Lo ordenamos de mayor a menor
#############################################################################
importances = rdf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rdf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Imprimimos las 19 caracteristicas más importantes
print("Top 19 caracteristicas más importantes:")

labels = np.array(['a' for _ in range(19)], dtype=object)
for f in range(19):
    print("%d. Features %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], features[indices[f]]))
    labels[f] = features[indices[f]]

# Figura con los 19 caracteristicas más importantes
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.title("Top 19 caracteristicas")
plt.bar(range(19), importances[indices[:19]], color="r", align="center")
plt.xticks(rotation=90)
plt.xticks(range(19), labels[:19])
plt.xlim([-1, 19])
plt.tight_layout()
plt.show()
print ()


###############################################################################
# Número de árboles
###############################################################################

# Probamos de 1 a 150, de 10 en 10
param_range = np.arange(1, 150, 10)


# Vamos a calcular la precision en el conjunto de entrenamiento y test por cada uno de los parametros
train_scores, test_scores = validation_curve(RandomForestClassifier(n_estimators=100,  random_state=2019),
                                             X_train,
                                             y_train,
                                             param_name="n_estimators",
                                             param_range=param_range,
                                             cv=3,
                                             scoring="accuracy")

# Conjunto de entrenamiento
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Conjunto de test
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print ("Conjunto de entrenamiento")
print (train_scores)

print ("\nConjunto de validación")
print (test_scores)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.plot(param_range, train_mean, label="Score del Conjunto de entrenamiento", color="red")
plt.plot(param_range, test_mean, label="Score de validación", color="blue")
plt.title("Precisión con respecto al número de arboles")
plt.xlabel("Número de arboles")
plt.ylabel("Precisión")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

####################################################################
# Volumen de datos
####################################################################

# Obtenemos los resultados de Cross-validation  para el conjunto de entrenamiento y test para
# varios tamaños
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(n_estimators=15,  random_state=2019),
                                                        X_train,
                                                        y_train,
                                                        cv=3,
                                                        scoring='accuracy',
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
# Conjunto de entrenamiento
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Conjunto de test
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.plot(train_sizes, train_mean, label="Precisión del Conjunto de entrenamiento", color="red")
plt.plot(train_sizes, test_mean,  label="Precisión del Conjunto de validación", color="blue")
plt.title("Curva de aprendizaje")
plt.xlabel("Tamaño del Conjunto de entrenamiento"), plt.ylabel("Precisión"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
