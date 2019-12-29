###################################################################################
# Clasificador: neural
#
# Autor: Carlos Muñiz Solaz
# Fecha: Enero 2020
#
# Utiliza clasificador red neuronal para clasificar el data set de entrenamiento de NB 15
###################################################################################

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sn
from pandas.api.types import CategoricalDtype


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from keras.wrappers.scikit_learn import  KerasClassifier
from time import time

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

attacks_types = CategoricalDtype( categories = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms'], ordered=True)
data.attack_cat = data.attack_cat.astype(attacks_types).cat.codes
y = data.iloc[:,-1].values


#####################################################################################################
# Descomponemos en los conjuntos de training y de test
#####################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  random_state=2019, stratify=y)

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

# Número de épocas de entrenamiento: probaremos valores entre 10 y 200
# Número de neuronas de la capa oculta: probaremos valores entre 20 y 1000.
# Velocidad de aprendizaje (learning rate): probaremos valores entre 0.0001 y 0.2.



rand_list = dict(epochs=[20, 50, 200],
                 num_hidden=np.random.randint(20, 500, size=10),
                 learning_rate=stats.uniform(0.001, 0.3))



#rand_list = dict(epochs=[20], #epochs=[10, 50, 150],
#                 num_hidden=np.random.randint(20, 1000, size=5),
#                 learning_rate=stats.uniform(0.0001, 0.2))

###rand_list = dict(epochs=[10],
###                 num_hidden=[200],
###                 learning_rate=[0.2])


print("Hiperparámetros: ")
print(rand_list)


#####################################################################################################
# Validación cruzada con 4 particiones estratificadas
#####################################################################################################
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2019)

#####################################################################################################
#  Modelo
#####################################################################################################

n_classes = 10


# creamos el modelo
def create_model(num_hidden=5, learning_rate=0.01):
    model = Sequential()
    # Capa de entrada
    model.add(Dense(num_hidden, activation='relu', input_dim = 19))
    # Capa oculta
    model.add(Dense(num_hidden, activation='sigmoid'))
    # Capa de salida
    model.add(Dense(n_classes, activation='softmax'))

    # la función de coste categorical_crossentropy es la más usada en clasificación multiclase
    # El sgd, stocastic gradiente descent, permite ajustar el factor de aprendizaje
    sgd = optimizers.SGD(lr=learning_rate)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc', 'mse'])

    return model


# Wrapper para poder llamar a modelo de Keras desde scikit
nnc = KerasClassifier(build_fn=create_model, batch_size=100, verbose=False)

#####################################################################################################
# Grid para explorar el espacio de parámetros
#####################################################################################################

start = time()
rand_search = RandomizedSearchCV(nnc, rand_list, cv=kfold, scoring='accuracy', n_iter=20)


# Buscamos el mejor modelo
rand_search.fit(X_train, y_train, verbose=True, shuffle=True)

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
best_epochs = rand_search.best_params_['epochs']
best_num_hidden = rand_search.best_params_['num_hidden']
best_learning_rate = rand_search.best_params_['learning_rate']


print ("Resumen del mejor modelo: ")
neural_model = create_model(num_hidden=best_num_hidden, learning_rate=best_learning_rate)
neural_model.summary()

#####################################################################################################
# Evaluamos el mejor modelo
#####################################################################################################

# One-hot encoded
y_train_encoded = keras.utils.to_categorical(y_train, n_classes)
y_test_encoded = keras.utils.to_categorical(y_test, n_classes)

start = time()
nn_score = neural_model.fit(X_train, y_train_encoded, epochs=best_epochs, batch_size=100, validation_data=(X_test, y_test_encoded), verbose=False, shuffle=True)
print("Tiempo para entrenar el mejor modelo %.2f seconds:" % ((time() - start)))

score_training = neural_model.evaluate(X_train, y_train_encoded, verbose = False)
print('Precisión del conjunto de entrenamiento:', score_training[1])

score_test = neural_model.evaluate(X_test, y_test_encoded, verbose = False)
print('Precisión del conjunto de test:', score_test[1])

# Vemos que tal predicce el conjunto de test
nn_pred = neural_model.predict(X_test, batch_size=100)
y_true = np.argmax(y_test_encoded, axis=1)
y_pred = np.argmax(nn_pred, axis=1)



#####################################################################################################
# Creamos la matriz de confusión
#####################################################################################################


print ("Matriz de confusión: ")
nn_cm = confusion_matrix(y_true, y_pred)
print()
print (nn_cm)


attacks_labels = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']

# Informe de clasificación
print ("\nInforme de clasificación: ")
print(classification_report(y_true, y_pred, labels = attacks_labels))

print(classification_report(y_true, y_pred))

#####################################################################################################
# Visualizamos la matriz de confusión
#####################################################################################################

nn_df_cm = pd.DataFrame(nn_cm, range(n_classes), range(n_classes))
plt.figure(figsize = (20,14))
sn.set(font_scale=1.4)
#sn.heatmap(nn_df_cm, annot=True, annot_kws={"size": 12}, xticklabels=True, yticklabels=True)

#attacks = ['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic', 'Normal', 'Reconnaissance','Shellcode','Worms']
sn.heatmap(nn_df_cm,  annot=True, annot_kws={"size": 10}, fmt='g', xticklabels=attacks_labels, yticklabels=attacks_labels)
sn.set(font_scale=1) # font size 2
plt.show()


#####################################################################################################
# Computamos las curvas ROC y sus areas para cada clase
#####################################################################################################
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], nn_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#####################################################################################################
# Dibujamos las curvas ROC y sus areas para cada clase
#####################################################################################################
plt.figure(1)
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='Curva ROC para la clase {0} (area = {1:0.2f})'''.format(attacks_labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Porcentaje de Falsos Positivos')
plt.ylabel('Porcentaje de Verdaderos Positivos')
plt.title('Curvas ROC de los distintos tipos de ataques')
plt.legend(loc="lower right")
plt.show()


# https://androidkt.com/get-the-roc-curve-and-auc-for-keras-model/
#uc_score=auc(y_pred,y_true)  #0.8822
#print (uc_score)


#####################################################################################################
# Grafica de como influyen los parametros en la precision del modelo
#####################################################################################################


plt.figure(0)

plt.plot(nn_score.history['acc'],'r')
plt.plot(nn_score.history['val_acc'],'g')
# plt.xticks(np.arange(0, 11, 2.0))
#plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Test Accuracy")
plt.legend(['training','test'])

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(nn_score.history['loss'],'r')
plt.plot(nn_score.history['val_loss'],'g')
#plt.xticks(np.arange(0, 11, 2.0))
#plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Test Loss")
plt.legend(['training','test'])


#plt.figure(3)
plt.subplot(1, 2, 2)
plt.plot(nn_score.history['mse'],'r')
plt.plot(nn_score.history['val_mse'],'g')
# plt.xticks(np.arange(0, 11, 2.0))
#plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("mse")
plt.title("Training mse vs Test mse")
plt.legend(['training','test'])

plt.show()
