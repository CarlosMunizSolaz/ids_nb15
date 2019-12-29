###################################################################################
# Clasificador: create_datasets
#
# Autor: Carlos Muñiz Solaz
# Fecha: Enero 2020
#
# Scripts para crear los data sets para los clasificadores clasicos
###################################################################################

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

df = pd.read_csv('/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_training_subset.csv', delimiter=",")
df = df.sample(frac=1).reset_index(drop=True)

print (df)


print ("To file /home/carlos/Projects/tfm/nb15/working/training_subset_rand.csv")

df.to_csv("/home/carlos/Projects/tfm/nb15/working/training_subset_rand.csv",  sep=",", header= True, float_format='%.10g', index=False)


attacks_types = CategoricalDtype( categories = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms'], ordered=True)
df.attack_cat = df.attack_cat.astype(attacks_types).cat.codes

df.to_csv("/home/carlos/Projects/tfm/nb15/working/training_subset_rand_numerical.csv",  sep=",", header= True, float_format='%.10g', index=False)






#############################################


df2 = pd.read_csv('/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_testing_subset.csv', delimiter=",")
df2 = df2.sample(frac=1).reset_index(drop=True)

print (df2)


print ("To file /home/carlos/Projects/tfm/nb15/working/testing_subset_rand.csv")

df2.to_csv("/home/carlos/Projects/tfm/nb15/working/testing_subset_rand.csv",  sep=",", header= True, float_format='%.10g', index=False)


attacks_types = CategoricalDtype( categories = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms'], ordered=True)
df2.attack_cat = df2.attack_cat.astype(attacks_types).cat.codes

df2.to_csv("/home/carlos/Projects/tfm/nb15/working/testing_subset_rand_numerical.csv",  sep=",", header= True, float_format='%.10g', index=False)


