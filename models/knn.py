###################################################################################
# Clasificador: kNN
#
# Autor: Carlos Muñiz Solaz
# Fecha: Enero 2020
#
# Utiliza clasificador k vecinos más proximos para clasificar el data set de entrenamiento de NB 15
###################################################################################

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sn
from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Qt5Agg')

np.random.seed(2019)  # Para garantizar la reproducibilidad
random.seed(2019)

# Función que permite contar el número de etiquetas
def proporcion_etiquetas(y):
    _, count = np.unique(y, return_counts=True)
    return np.round(np.true_divide(count, y.shape[0]) * 100, 2)


#####################################################################################################
# Leemos el fichero de entrada
#####################################################################################################
#data = pd.read_csv('/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_training_subset.csv')
data = pd.read_csv('/home/carlos/Projects/tfm/nb15/working//training_subset_rand.csv')

X = data.iloc[:,0:19]


y = data.iloc[:,-1]
labels = y  # Clasificación de los ataques
features = list ( data.columns.values[0:19] ) # Nombre de las columnas

#####################################################################################################
# Descomponemos en los conjuntos de training y de test
#####################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2019, stratify=y)

# Normalizamos
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

print("Número de registros para entrenar: {}".format(X_train.shape[0]))
print("Número de registros para test: {}".format(X_test.shape[0]))


print("\nProporción de las etiquetas en el conjunto original: {}".format(proporcion_etiquetas(labels)))
print("Proporción de las etiquetas en el conjunto de entrenamiento: {}".format(proporcion_etiquetas(y_train)))
print("Proporción de las etiquetas en el conjunto de test: {}".format(proporcion_etiquetas(y_test)))


#####################################################################################################
# Selección de hiperparámetros óptimos
#####################################################################################################
k_range = list(range(1, 11))  # Entre 1 y 10
weight_options = ['uniform', 'distance']


param_grid = dict(n_neighbors=k_range, weights=weight_options)

print("\nHiperparámetros: ")
print(param_grid)


#####################################################################################################
# Validación cruzada con 4 particiones estratificadas
#####################################################################################################
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2019)

#####################################################################################################
#  Modelo
#####################################################################################################
knn = KNeighborsClassifier()

#####################################################################################################
# Grid para explorar el espacio de parámetros
#####################################################################################################
start = time()
grid = GridSearchCV(knn, param_grid, cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)
print("Tiempo para encontrar el mejor modelo %.2f seconds:" % ((time() - start)))

#####################################################################################################
# Guardamos y mostramos los resultados
#####################################################################################################
results = grid.cv_results_

# Mostramos los resultados
print("\nResultados: ")
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print("\nLos mejores parámetros para el conjunto de entrenamiento son {}".format(grid.best_params_))
print("\nPrecisión cross-validation para el conjunto de entrenamiento: {}".format(grid.best_score_))
print("Precisión cross-validation para el conjunto de test: {}".format(grid.score(X_test, y_test)))

print ("\n")
print(grid.best_estimator_)

#####################################################################################################
# Estudiamos el mejor modelo
#####################################################################################################

start = time()

best_n_neighbors = grid.best_params_['n_neighbors']
best_weights = grid.best_params_['weights']

knn = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights)

# Entrenamos al knn
knn.fit(X_train, y_train)

print("Tiempo para entrenar al mejor modelo %.2f seconds:" % ((time() - start)))

# Comprobamos la precision en el conjunto de entrenamiento y de test
print ("Precisión del conjunto de entrenamiento:", knn.score(X_train, y_train))
print ("Precisión del conjunto de test: ", knn.score(X_test, y_test))


# Vemos que tal predice el conjunto de test
y_true, y_pred = y_test, knn.predict(X_test)

print ("Matriz de confusión: ")
mc = confusion_matrix(y_true, y_pred)
print (mc)


attacks_labels = ['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic', 'Normal', 'Reconnaissance','Shellcode','Worms']



mc_df_cm = pd.DataFrame(mc, range(10), range(10))

plt.figure(figsize = (20,14))
sn.set(font_scale=1.4)

sn.heatmap(mc_df_cm,  annot=True, annot_kws={"size": 12}, fmt='g', xticklabels=attacks_labels, yticklabels=attacks_labels)
sn.set(font_scale=1) # font size 2

plt.show()

# Informe de clasificación
print ("\nInforme de clasificación: ")
print(classification_report(y_true, y_pred))


#####################################################################################################
# Grafica de como influye K en el accuracy del modelo
#####################################################################################################
train_accuracy = []
test_accuracy = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, weights='distance')
    knn.fit(X_train, y_train)
    # Precisión de entrenamiento
    train_accuracy.append(knn.score(X_train, y_train))
    # Precisión de test
    test_accuracy.append(knn.score(X_test, y_test))

#plt.figure(figsize=[13,8])
plt.subplot(1, 2, 1)
plt.plot(k_range, test_accuracy, label = 'Precisión en Testing')
plt.plot(k_range, train_accuracy, label = 'Precisión en Entrenamiento')
plt.legend()
plt.title('Valores de k - Precisión (pesos = distance)')
plt.xlabel('Número de vecinos')
plt.ylabel('Precisión')
plt.xticks(k_range)
plt.tight_layout()
print("La mejor precisión en el conjunto de test es {} con K = {}".format(np.max(test_accuracy),
                                                                          1+test_accuracy.index(np.max(test_accuracy))))


#plt.show()

#####################################################################################################
# Grafica de como influye  weights en el accuracy del modelo
#####################################################################################################
train_accuracy = []
test_accuracy = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, weights='uniform')
    knn.fit(X_train, y_train)
    # Precisión de entrenamiento
    train_accuracy.append(knn.score(X_train, y_train))
    # Precisión de test
    test_accuracy.append(knn.score(X_test, y_test))

#plt.figure(figsize=[13,8])
plt.subplot(1, 2, 2)
plt.plot(k_range, test_accuracy, label = 'Precisión en Testing')
plt.plot(k_range, train_accuracy, label = 'Precisión en Entrenamiento')
plt.legend()
plt.title('Valores de k - Precisión (pesos = uniform)')
plt.xlabel('Número de vecinos')
plt.ylabel('Precisión')
plt.xticks(k_range)
plt.tight_layout()
print("La mejor precisión en el conjunto de test es {} con K = {}".format(np.max(test_accuracy),
                                                                          1+test_accuracy.index(np.max(test_accuracy))))

plt.show()
