###################################################################################
# Clasificador: hierarchical
#
# Autor: Carlos Muñiz Solaz
# Fecha: Enero 2020
#
# Utiliza clasificador jerarquico de random forests para clasificar el data set de validación de NB 15
###################################################################################

import numpy as np
import random
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from time import time

import matplotlib.pyplot as plt
import seaborn as sn

import matplotlib
matplotlib.use('Qt5Agg')

np.random.seed(2019)  # Para garantizar la reproducibilidad
random.seed(2019)


# Funcion que permite contar el numero de etiquetas
def proporcion_etiquetas(y):
    _, count = np.unique(y, return_counts=True)
    return np.round(np.true_divide(count, y.shape[0]) * 100, 2)




data = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_total_numeric_testing_WR_19FS-60percent.csv", delimiter=",", skiprows=1)
X_train = data[:,0:19]
# Normalizamos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
###############################################


#Starting time of Model Creation
start_creation = time()

#################################################################################################
# Level 1
##################################################################################################
start = time()

# Clasificación ataque o no ataque en entrenamiento
# Nivel 1
data = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/1or0.csv",delimiter=",", skiprows=1)
X_train = data[:,0:19]

X_train = sc.transform(X_train)
y_train = data[:,-1]
rf1 = RandomForestClassifier(n_estimators=15, class_weight="balanced", random_state=2019) #  class_weight="balanced",
rf1 = rf1.fit(X_train, y_train)

# Precisión usando validación cruzada del conjunto de entrenamiento
scores = cross_val_score(rf1, X_train, y_train, cv = 4)
print("Training at Level 1: %.2f seconds:" % ((time() - start)))
print ('Cross-validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))


#####################################################################################################
# Computamos las curvas ROC y sus areas para cada clase
#####################################################################################################
y_pred = rf1.predict(X_train)

fpr, tpr, _ = roc_curve(y_train, y_pred)
roc_auc = auc(fpr, tpr)

#####################################################################################################
# Dibujamos las curvas ROC y sus areas para cada clase
#####################################################################################################

plt.plot(fpr, tpr, label='Curva ROC para la clase normal (area = {0:0.3f})'''.format(roc_auc))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Porcentaje de Falsos Positivos')
plt.ylabel('Porcentaje de Verdaderos Positivos')
plt.title('Curvas ROC de los distintos tipos de ataques')
plt.legend(loc="lower right")
plt.show()



#############################################

# Obtenemos los resultados de Cross-validation  para el conjunto de entrenamiento y test para
# varios tamaños
######### train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(n_estimators=100,  random_state=2019),
#########                                                        X_train,
#########                                                        y_train,
#########                                                        cv=3,
#########                                                        scoring='accuracy',
#########                                                        train_sizes=np.linspace(0.01, 1.0, 50))

#Conjunto de entrenamiento
######### train_mean = np.mean(train_scores, axis=1)
######### train_std = np.std(train_scores, axis=1)

# Conjunto de test
######### test_mean = np.mean(test_scores, axis=1)
######### test_std = np.std(test_scores, axis=1)

######### fig, ax = plt.subplots(1, 1, figsize=(8, 8))
######### plt.plot(train_sizes, train_mean, label="Precisión del Conjunto de entrenamiento", color="red")
######### plt.plot(train_sizes, test_mean,  label="Precisión del Conjunto de validación", color="blue")
######### plt.title("Curva de aprendizaje")
######### plt.xlabel("Tamaño del Conjunto de entrenamiento"), plt.ylabel("Precisión"), plt.legend(loc="best")
######### plt.tight_layout()
######### plt.show()




#################################################################################################
# Level 2
##################################################################################################
start = time()

# Dos/Exploits vs others
# Nivel 2
data = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/part1.csv",delimiter=",", skiprows=1)
X_train = data[:,0:19]
X_train = sc.transform(X_train)
y_train = data[:,-1]
rf2 = RandomForestClassifier(n_estimators=15, class_weight="balanced", random_state=2019) # class_weight="balanced"
rf2 = rf2.fit(X_train, y_train)

# Precisión usando validación cruzada del conjunto de entrenamiento
scores = cross_val_score(rf2, X_train, y_train, cv = 4)
print("Training at Level 2: %.2f seconds:" % ((time() - start)))
print ('Cross-validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))


#################################################################################################
# Level 3
##################################################################################################
# Nivel 3a

start = time()

# Clasificación de  DoS vs. Exploit
data = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/part2.csv",delimiter=",", skiprows=1)
X_train = data[:,0:19]
X_train = sc.transform(X_train)
y_train = data[:,-1]
rf3a = RandomForestClassifier(n_estimators=15, class_weight="balanced", random_state=2019)   # class_weight="balanced",
rf3a =  rf3a.fit(X_train, y_train)

# Precisión usando validación cruzada del conjunto de entrenamiento
scores = cross_val_score(rf3a, X_train, y_train, cv = 4)
print("Training at Level 3a: %.2f seconds:" % ((time() - start)))
print ('Cross-validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))

##################################################################################################
# Nivel 3b
start = time()

#  Clasificación de Resto de ataques
data=np.loadtxt("/home/carlos/Projects/tfm/nb15/working/part3.csv",delimiter=",", skiprows=1)
X_train = data[:,0:19]
X_train = sc.transform(X_train)
y_train = data[:,-1]
rf3b = RandomForestClassifier(n_estimators=15, class_weight="balanced", random_state=2019) # class_weight="balanced",
rf3b = rf3b.fit(X_train, y_train)


# Precisión usando validación cruzada del conjunto de entrenamiento
scores = cross_val_score(rf3b, X_train, y_train, cv = 4)
print("Training at Level 3b: %.2f seconds:" % ((time() - start)))
print ('Cross-validation Accuracy: %0.4f (+/- %0.4f)' % (scores.mean(), scores.std()))

#############################################################################################


# Tiempo de creación del modelo
print("Training %.2f seconds:" % ((time() - start_creation)))



start = time()
data = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_total_numeric_testing_WR_19FS-60percent.csv", delimiter=",", skiprows=1)
x = data[:,0:19]
x = sc.transform(x)

y = data[:,19]
result = np.zeros(len(y))
a = 0
knt=False

# print (x.shape)

# Three stages
for p in x:
    if a % 500 == 0:
        print (a)
    train = p[0:19]
    if (rf1.predict(train.reshape(1,-1)) == 0):
        result[a]=0
        a=a+1
        continue
    if rf2.predict(train.reshape(1,-1)) == 1:
        result[a]=rf3a.predict(train.reshape(1,-1))
    if rf2.predict(train.reshape(1,-1)) == 2:
        result[a]=rf3b.predict(train.reshape(1,-1))
    a=a+1

target_names = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']


print ()
conf=confusion_matrix(y,result)
print()
print(classification_report(y,result,target_names=target_names))
print ()
print (conf)
print ()
print ("Normal:    "+str(float(conf[0,0])/np.count_nonzero(y==0)))
print ("Analysis:  "+str(float(conf[1,1])/np.count_nonzero(y==1)))
print ("Backdoor:  "+str(float(conf[2,2])/np.count_nonzero(y==2)))
print ("DoS:       "+str(float(conf[3,3])/np.count_nonzero(y==3)))
print ("Exploit:   "+str(float(conf[4,4])/np.count_nonzero(y==4)))
print ("Fuzzers:   "+str(float(conf[5,5])/np.count_nonzero(y==5)))
print ("Generic:   "+str(float(conf[6,6])/np.count_nonzero(y==6)))
print ("Recon:     "+str(float(conf[7,7])/np.count_nonzero(y==7)))
print ("Shell:     "+str(float(conf[8,8])/np.count_nonzero(y==8)))
print ("Worms:     "+str(float(conf[9,9])/np.count_nonzero(y==9)))
print("Testing %.2f seconds:" % ((time() - start)))
print ('Training Accuracy : {}'.format(accuracy_score(y, result)))
print ()


#################################################################################################
# Fase de validación
##################################################################################################
start = time()
data = np.loadtxt("/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_total_numeric_testing_WR_19FS-40percent.csv", delimiter=",", skiprows=1)

x = data[:,0:19]
x = sc.transform(x)
y = data[:,19]
result = np.zeros(len(y))
a = 0
knt=False

# print (x.shape)

# Three stages
for p in x:
    if a % 500 == 0:
        print (a)
    test = p[0:19]
    if (rf1.predict(test.reshape(1,-1)) == 0):
        result[a]=0
        a=a+1
        continue
    if rf2.predict(test.reshape(1,-1)) == 1:
        result[a]=rf3a.predict(test.reshape(1,-1))
    if rf2.predict(test.reshape(1,-1)) == 2:
        result[a]=rf3b.predict(test.reshape(1,-1))
    a=a+1

target_names = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']


print ()
conf=confusion_matrix(y,result)
print()
print(classification_report(y,result,target_names=target_names))
print ()
print (conf)
print ()
print ("Normal:    "+str(float(conf[0,0])/np.count_nonzero(y==0)))
print ("Analysis:  "+str(float(conf[1,1])/np.count_nonzero(y==1)))
print ("Backdoor:  "+str(float(conf[2,2])/np.count_nonzero(y==2)))
print ("DoS:       "+str(float(conf[3,3])/np.count_nonzero(y==3)))
print ("Exploit:   "+str(float(conf[4,4])/np.count_nonzero(y==4)))
print ("Fuzzers:   "+str(float(conf[5,5])/np.count_nonzero(y==5)))
print ("Generic:   "+str(float(conf[6,6])/np.count_nonzero(y==6)))
print ("Recon:     "+str(float(conf[7,7])/np.count_nonzero(y==7)))
print ("Shell:     "+str(float(conf[8,8])/np.count_nonzero(y==8)))
print ("Worms:     "+str(float(conf[9,9])/np.count_nonzero(y==9)))
print("Testing %.2f seconds:" % ((time() - start)))
print ('Test Accuracy : {}'.format(accuracy_score(y, result)))
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


