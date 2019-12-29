# Detección de ataques en el entorno de Internet de las Cosas

**Carlos Muñiz Solaz**  
Máster Universitario de Ciencia de Datos de la UOC    
Área 5.1 - Detección de Patrones  

**Consultor:** Carlos Hernández Gañán  
**Profesor Responsable de la Asignatura:** Albert Solé Ribalta  

Enero 2020

---  

Este repositorio contiene todo el código implementado durante la elaboración del Trabajo Fin de Máster. 

Debido al desarrollo que se ha producido en los últimos años en las tecnologias de red y en los entornos IoT, se ha producido un aumento en el número de cyber-ataques y amenazas. Muchos insvestigadores de cyber seguridad llevan a cabo estudios para proteger estos sistemas.

En este repositorio se ha subido el código y los data sets empleados para la construcción de sistemas de detección de intrusos usando técnicas de **aprendizaje automático (machine learning)**.

Los detalles del data set empleado están disponibles en el siguiente articulo:

*Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)."*

La **página oficial** del data set se puede encontrar aqui:  
https://www.unsw.adfa.edu.au/australian-centre-for-cyber-security/cybersecurity/ADFA-NB15-Datasets/

La memoria del trabajo está disponible en la biblioteca digital de la Universidad Oberta de Catalunya.

# Estructura del Repositorio

* **datasets:**
  - **UNSW_NB15_testing_subset.csv:** Data set original de validación NB15 
  - **UNSW_NB15_training_subset.csv:** Data set original de entrenamiento NB15 
  - **UNSW_NB15_testing_subset.arff:** Data set original de validación NB15 en formato Weka
  - **UNSW_NB15_training_subset.arff:** Data set original de entrenamiento NB15 en formato Weka
  - **testing_subset_rand.csv:** Data set permutado de validación NB15 (no se usa)
  - **testing_subset_rand_numerical.csv:** Data set permutado de validación NB15 con las clases numéricas (no se usa)
  - **training_subset_rand.csv:** **Data set permutado de entrenamiento NB15. Usado por los algoritmos de aprendizaje clasico**
  - **training_subset_rand_numerical.csv:** Data set permutado de entrenamiento NB15 con las clases numéricas (no se usa)
* **datasets/hierarchical:**
  - **40percent.csv:** 	40 % data set original permutado de entrenamiento NB15
  - **60percent.csv:** 	60 % data set original permutado de entrenamiento NB15. A partir de este data set se construyeron los siguientes.
  - **Nivel1.csv:** data set con ataques y no ataques (binario) 
  - **Nivel2.csv:** data set con ataques DoS/Exploits y el resto de ataques
  - **Nivel3a.csv:** data set solo con ataques DoS y Exploits
  - **Nivel3b.csv:** data set con ataques que no son DoS o Exploits

* models:
* scripts:
