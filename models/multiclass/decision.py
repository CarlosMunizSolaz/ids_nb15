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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
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
sample_split_range =  list(range(2, 52, 5))
max_depth_range =  list(range(2, 52, 5))

param_grid = dict(min_samples_split=sample_split_range, max_depth=max_depth_range)

print("\nHiperparámetros: ")
print(param_grid)


#####################################################################################################
# Validación cruzada con 4 particiones estratificadas
#####################################################################################################
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2019)

#####################################################################################################
#  Modelo
#####################################################################################################
dtc = DecisionTreeClassifier()

#####################################################################################################
# Grid para explorar el espacio de parámetros
#####################################################################################################
start = time()
grid = GridSearchCV(dtc, param_grid, cv=kfold, scoring='accuracy')
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

best_min_samples_split = grid.best_params_['min_samples_split']
best_max_depth = grid.best_params_['max_depth']

dtc = DecisionTreeClassifier(min_samples_split=best_min_samples_split, max_depth=best_max_depth)

start = time()
# Entrenamos al dtc
decision_tree = dtc.fit(X_train, y_train)
print("Tiempo para entrenar el mejor modelo %.2f seconds:" % ((time() - start)))


# Imprimimos el dtc
r = export_text(decision_tree, feature_names=features)
print (r)

# Comprobamos la precision en el conjunto de entrenamiento y de test
print ("Precisión del conjunto de entrenamiento:", dtc.score(X_train, y_train))
print ("Precisión del conjunto de test: ", dtc.score(X_test, y_test))


# Vemos que tal predicce el conjunto de test
y_true, y_pred = y_test, dtc.predict(X_test)

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
# Grafica de como influye el min _samples_split en el accuracy del modelo
#####################################################################################################
train_accuracy = []
test_accuracy = []
for sample in sample_split_range:
    dtc = DecisionTreeClassifier(min_samples_split = sample, max_depth=best_max_depth)
    dtc.fit(X_train, y_train)
    # Precisión de entrenamiento
    train_accuracy.append(dtc.score(X_train, y_train))
    # Precisión de test
    test_accuracy.append(dtc.score(X_test, y_test))

plt.figure(figsize=[13,8])
plt.plot(sample_split_range, test_accuracy, label = 'Precisión en Testing')
plt.plot(sample_split_range, train_accuracy, label = 'Precisión en Entrenamiento')
plt.legend()
plt.title('Valores de sample_split - Precisión')
plt.xlabel('Número de sample_split')
plt.ylabel('Precisión')
plt.xticks(sample_split_range)
plt.tight_layout()
print("La mejor precisión en el conjunto de test es {} con sample_split_range = {}".format(np.max(test_accuracy),
                                                                          1+test_accuracy.index(np.max(test_accuracy))))
plt.show()

#####################################################################################################
# Grafica de como influye el min max_depth en el accuracy del modelo
#####################################################################################################
train_accuracy = []
test_accuracy = []
for depth in max_depth_range:
    dtc = DecisionTreeClassifier(min_samples_split = best_min_samples_split, max_depth=depth)
    dtc.fit(X_train, y_train)
    # Precisión de entrenamiento
    train_accuracy.append(dtc.score(X_train, y_train))
    # Precisión de test
    test_accuracy.append(dtc.score(X_test, y_test))

plt.figure(figsize=[13,8])
plt.plot(sample_split_range, test_accuracy, label = 'Precisión en Testing')
plt.plot(sample_split_range, train_accuracy, label = 'Precisión en Entrenamiento')
plt.legend()
plt.title('Valores de Profundidad del árbol - Precisión')
plt.xlabel('Depth')
plt.ylabel('Precisión')
plt.xticks(max_depth_range)
plt.tight_layout()
print("La mejor precisión en el conjunto de test es {} con depth = {}".format(np.max(test_accuracy),
                                                                          1+test_accuracy.index(np.max(test_accuracy))))

plt.show()
