import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from time import time

from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


import matplotlib
matplotlib.use('Qt5Agg')

np.random.seed(2019)  # Para garantizar la reproducibilidad
random.seed(2019)

# Funcion que permite contar el numero de etiquetas
def proporcion_etiquetas(y):
    _, count = np.unique(y, return_counts=True)
    return np.round(np.true_divide(count, y.shape[0]) * 100, 2)


#####################################################################################################
# Leemos el fichero de entrada
#####################################################################################################

#data = pd.read_csv('/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_training_subset.csv')
data = pd.read_csv('/home/carlos/Projects/tfm/nb15/working//training_subset_rand.csv')

X = data.iloc[:,0:19]
y = data.iloc[:,19]
labels = y  # Clasificación de los ataques
features = list ( data.columns.values[0:19] ) # Nombre de las columnas

#####################################################################################################
# Descomponemos en los conjuntos de training y de test
#####################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2019, stratify=y)

# Normalizamos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

print("Número de registros para entrenar: {}".format(X_train.shape[0]))
print("Número de registros para test: {}".format(X_test.shape[0]))

print("\nProporción de las etiquetas en el conjunto original: {}".format(proporcion_etiquetas(labels)))
print("Proporción de las etiquetas en el conjunto de entrenamiento: {}".format(proporcion_etiquetas(y_train)))
print("Proporción de las etiquetas en el conjunto de test: {}".format(proporcion_etiquetas(y_test)))


#####################################################################################################
# Selección de hiperparámetros óptimos
#####################################################################################################
# Seleccionamos los parámetros
rand_list = dict(C=stats.uniform(1, 500),  gamma=stats.uniform(10e-9, 10e-3))

print("\nHiperparámetros: ")
print(rand_list)


#####################################################################################################
# Validación cruzada con 4 particiones estratificadas
#####################################################################################################
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2019)

#####################################################################################################
#  Modelo
#####################################################################################################
# Modelo SVM con radial base function
svm = SVC(kernel='rbf')

#####################################################################################################
# Grid para explorar el espacio de parámetros
#####################################################################################################
#rand_search = RandomizedSearchCV(svm, param_distributions=rand_list, n_iter = 10, cv = kfold, random_state = 2019, scoring = 'accuracy')

start = time()
rand_search = RandomizedSearchCV(svm, param_distributions=rand_list, n_iter = 20, cv = kfold, random_state = 2019, scoring = 'accuracy', verbose= True)
rand_search.fit(X_train, y_train)

print("Tiempo para encontrar el mejor modelo %.2f seconds:" % ((time() - start)))

#####################################################################################################
# Guardamos y mostramos los resultados
#####################################################################################################
results = rand_search.cv_results_

# Mostramos los resultados
print("\nResultados: ")
means = rand_search.cv_results_['mean_test_score']
stds = rand_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, rand_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print("\nLos mejores parámetros para el conjunto de entrenamiento son {}".format(rand_search.best_params_))
print("\nPrecisión cross-validation para el conjunto de entrenamiento: {}".format(rand_search.best_score_))
print("Precisión cross-validation para el conjunto de test: {}".format(rand_search.score(X_test, y_test)))

print ("\n")
print(rand_search.best_estimator_)

#####################################################################################################
# Estudiamos el mejor modelo
#####################################################################################################

best_C = rand_search.best_params_['C']
best_gamma = rand_search.best_params_['gamma']

start = time()
svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma)

# Entrenamos al knn
svm.fit(X_train, y_train)

print("Tiempo para entrenar el mejor modelo %.2f seconds:" % ((time() - start)))


# Comprobamos la precision en el conjunto de entrenamiento y de test
print ("Precisión del conjunto de entrenamiento:", svm.score(X_train, y_train))
print ("Precisión del conjunto de test: ", svm.score(X_test, y_test))


# Vemos que tal predicce el conjunto de test
y_true, y_pred = y_test, svm.predict(X_test)

print ("Matriz de confusión: ")
cm = confusion_matrix(y_true, y_pred)
print (cm)

# Informe de clasificación
print ("\nInforme de clasificación: ")
print(classification_report(y_true, y_pred))


#####################################################################################################
# Visualizamos la matriz de confusión
#####################################################################################################
n_classes = 10
df_cm = pd.DataFrame(cm, range(n_classes), range(n_classes))
plt.figure(figsize = (20,14))

attacks_labels = ['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic', 'Normal', 'Reconnaissance','Shellcode','Worms']
sn.set(font_scale=1.4)
sn.heatmap(df_cm,  annot=True, annot_kws={"size": 10}, fmt='g', xticklabels=attacks_labels, yticklabels=attacks_labels)
sn.set(font_scale=1) # font size 2
plt.show()


#####################################################################################################
# Grafica de como influye el C en el accuracy del modelo
#####################################################################################################
train_accuracy = []
test_accuracy = []


C=[30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
for c in  C:
    svm = SVC(kernel='rbf', C=c, gamma=best_gamma)
    svm.fit(X_train, y_train)
    # Precisión de entrenamiento
    train_accuracy.append(svm.score(X_train, y_train))
    # Precisión de test
    test_accuracy.append(svm.score(X_test, y_test))

plt.figure(figsize=[13,8])
plt.plot(C, test_accuracy, label = 'Precisión en Testing')
plt.plot(C, train_accuracy, label = 'Precisión en Entrenamiento')
plt.legend()
plt.title('Valores de C - Precisión')
plt.xlabel('C')
plt.ylabel('Precisión')
plt.xticks(C)
plt.tight_layout()
print("La mejor precisión en el conjunto de test es {} con C = {}".format(np.max(test_accuracy),
                                                                          1+test_accuracy.index(np.max(test_accuracy))))
plt.show()

#####################################################################################################
# Grafica de como influye el ammas en el accuracy del modelo
#####################################################################################################
train_accuracy = []
test_accuracy = []
gammas= [1e-9, 1e-6, 1e-3,  1e3, 1e6, 1e9]
for gamma2 in gammas:
    svm = SVC(kernel='rbf', C=best_C, gamma=gamma2)
    svm.fit(X_train, y_train)
    # Precisión de entrenamiento
    train_accuracy.append(svm.score(X_train, y_train))
    # Precisión de test
    test_accuracy.append(svm.score(X_test, y_test))

plt.figure(figsize=[13,8])
plt.plot(gammas, test_accuracy, label = 'Precisión en Testing')
plt.plot(gammas, train_accuracy, label = 'Precisión en Entrenamiento')
plt.legend()
plt.title('Valores de gammas - Precisión')
plt.xlabel('Gamma')
plt.ylabel('Precisión')
plt.xticks(gammas)
plt.tight_layout()
print("La mejor precisión en el conjunto de test es {} con depth = {}".format(np.max(test_accuracy),
                                                                          1+test_accuracy.index(np.max(test_accuracy))))

plt.show()


