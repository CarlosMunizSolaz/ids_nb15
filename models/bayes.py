###################################################################################
# Clasificador: Bayes
#
# Autor: Carlos Muñiz Solaz
# Fecha: Enero 2020
#
# Utiliza clasificador Naive Bayes para clasificar el data set de entrenamiento de NB 15
###################################################################################

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

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
#data = pd.read_csv('/home/carlos/Projects/tfm/nb15/working//testing_subset_rand.csv')
print (data.columns.values[0:20])

ax = data.groupby('attack_cat').size().plot(kind='bar', figsize=(8,8), color=['black', 'red', 'green', 'blue', 'cyan', 'purple', 'grey', 'brown', 'yellow' ],  edgecolor='black')
attacks = ['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic', 'Normal', 'Reconnaissance','Shellcode','Worms']
# ax.set_xticklabels(attacks, rotation = 1)
ax.set_ylabel ('Número de observaciones')
ax.set_xlabel ('Categoría')
ax.set_title ('Número de observaciones por categoría')
plt.figure()
plt.show()




data = pd.read_csv('/home/carlos/Projects/tfm/nb15/working//testing_subset_rand.csv')
#data = pd.read_csv('/home/carlos/Projects/tfm/nb15/working//testing_subset_rand.csv')
print (data.columns.values[0:20])

ax = data.groupby('attack_cat').size().plot(kind='bar', figsize=(8,8), color=['black', 'red', 'green', 'blue', 'cyan', 'purple', 'grey', 'brown', 'yellow' ],  edgecolor='black')
attacks = ['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic', 'Normal', 'Reconnaissance','Shellcode','Worms']
# ax.set_xticklabels(attacks, rotation = 1)
ax.set_ylabel ('Número de observaciones')
ax.set_xlabel ('Categoría')
ax.set_title ('Número de observaciones por categoría')
plt.figure()
plt.show()
exit()
#data = pd.read_csv('/home/carlos/Projects/tfm/nb15/working//training_subset_rand_numerical.csv')

# Create correlation matrix
corr_matrix = data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]# 0.95)]



###data= data.drop(data[to_drop], axis=1)

print (to_drop)
###print (data.shape)

print (type (data))


data = data.replace( ['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic', 'Reconnaissance','Shellcode','Worms'], 'Attack')
print (data)
#exit()
#X = data.iloc[:,0:15]

#X = data.iloc[:,0:19]
X = data.iloc[:,0:12]

# X = data.iloc[:,0:19]

y = data.iloc[:,-1]

print (y)
labels = y  # Clasificación de los ataques
# features = list ( data.columns.values[0:19] ) # Nombre de las columnas

#print (features)

#####################################################################################################
# Descomponemos en los conjuntos de training y de test
#####################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2019, stratify=y)

#from sklearn.preprocessing import StandardScaler
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
var_smoothing_rate = dict(var_smoothing_rate=stats.uniform(1e-9, 1e-9)) #[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3] #stats.uniform(1e-9, 1e-9)



param_grid = dict(var_smoothing=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e3, 1e6, 1e9, 1e15])

print("\nHiperparámetros: ")
print(param_grid)


#####################################################################################################
# Validación cruzada con 4 particiones estratificadas
#####################################################################################################
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2019)

#####################################################################################################
#  Modelo
#####################################################################################################
gnbc = GaussianNB()

#####################################################################################################
# Grid para explorar el espacio de parámetros
#####################################################################################################
grid = GridSearchCV(gnbc, param_grid, cv=kfold, scoring='accuracy')

print (X_train)
print (y_train)
grid.fit(X_train, y_train)

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

from sklearn.metrics import accuracy_score
BernNB = BernoulliNB(binarize=1.0)
BernNB.fit(X_train,y_train)
print(BernNB)
y_expect = y_test
y_predict = BernNB.predict(X_test)
print (accuracy_score(y_expect,y_predict))

#####################################################################################################
# Estudiamos el mejor modelo
#####################################################################################################

best_var_smoothing = grid.best_params_['var_smoothing']


gnbc = GaussianNB(var_smoothing=best_var_smoothing)

# Entrenamos al gbc
gnbc.fit(X_train, y_train)

# Comprobamos la precision en el conjunto de entrenamiento y de test
print ("Precisión del conjunto de entrenamiento:", gnbc.score(X_train, y_train))
print ("Precisión del conjunto de test: ", gnbc.score(X_test, y_test))


# Vemos que tal predicce el conjunto de test
y_true, y_pred = y_test, gnbc.predict(X_test)

print ("Matriz de confusión: ")
print (confusion_matrix(y_true, y_pred))

# Informe de clasificación
print ("\nInforme de clasificación: ")
print(classification_report(y_true, y_pred))

exit()
#####################################################################################################
# Grafica de como influye el smoothing en el accuracy del modelo
#####################################################################################################
train_accuracy = []
test_accuracy = []

for smoothing in var_smoothing_rate:
    gnbc = GaussianNB(var_smoothing=smoothing)
    gnbc.fit(X_train, y_train)
    # Precisión de entrenamiento
    train_accuracy.append(gnbc.score(X_train, y_train))
    # Precisión de test
    test_accuracy.append(gnbc.score(X_test, y_test))

plt.figure(figsize=[13,8])
plt.plot(var_smoothing_rate, test_accuracy, label = 'Precisión en Testing')
plt.plot(var_smoothing_rate, train_accuracy, label = 'Precisión en Entrenamiento')
plt.legend()
plt.title('Valores de smoothing - Precisión')
plt.xlabel('Parametro var_smoothing')
plt.ylabel('Precisión')
plt.xticks(var_smoothing_rate)
plt.tight_layout()
print("La mejor precisión en el conjunto de test es {} con var_smoothing = {}".format(np.max(test_accuracy),
                                                                          1+test_accuracy.index(np.max(test_accuracy))))


plt.show()

