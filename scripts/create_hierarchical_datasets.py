###################################################################################
# Clasificador: create_hierarchical_datasets
#
# Autor: Carlos Muñiz Solaz
# Fecha: Enero 2020
#
# Scripts para crear los data sets para los clasificadores jerarquicos
###################################################################################


import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

df = pd.read_csv('/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_testing_subset.csv', delimiter=",")
df = df.sample(frac=1).reset_index(drop=True)


df_60percent = df[df.index<=np.percentile(df.index, 60)].copy()
df_40percent = df[df.index > np.percentile(df.index, 60)].copy()


df_60percent.to_csv("/home/carlos/Projects/tfm/nb15/working/60percent_labels.csv",  sep=",", header= True, float_format='%.10g', index=False)
df_40percent.to_csv("/home/carlos/Projects/tfm/nb15/working/40percent_labels.csv",  sep=",", header= True, float_format='%.10g', index=False)

df_1o0 = df_60percent.copy()
df_1o0['attack_cat'] = np.where(df_1o0['attack_cat']=='Normal', '0', '1')

df_1o0.to_csv("/home/carlos/Projects/tfm/nb15/working/1or0.csv",  sep=",", header= True, float_format='%.10g', index=False)



#################################################################

df_part1 = df_60percent [df_60percent.index<=np.percentile(df_60percent.index, 33)].copy()

df_part1 = df_part1.drop(df_part1[ df_part1['attack_cat']=='Normal' ].index)

#df_part1['attack_cat'] = np.where(df_part1['attack_cat']=='Normal',
#                                '0', np.where( (df_part1['attack_cat']=='DoS') |
#                                               (df_part1['attack_cat']=='Exploits'), '1', '2'))

df_part1['attack_cat'] = np.where( (df_part1['attack_cat']=='DoS') |
                                   (df_part1['attack_cat']=='Exploits'), '1', '2')

df_part1.to_csv("/home/carlos/Projects/tfm/nb15/working/part1.csv",  sep=",", header= True, float_format='%.10g', index=False)



#################################################################

df_part2 = df_60percent [(df_60percent.index>np.percentile(df_60percent.index, 33)) & (df_60percent.index<=np.percentile(df_60percent.index, 66))].copy()


df_part2 = df_part2.drop( df_part2[ (df_part2['attack_cat']!='DoS') &  (df_part2['attack_cat']!='Exploits') ].index)

df_part2['attack_cat'] = np.where( (df_part2['attack_cat']=='DoS'), '3', '4')


df_part2.to_csv("/home/carlos/Projects/tfm/nb15/working/part2.csv",  sep=",", header= True, float_format='%.10g', index=False)


#################################################################



df_part3 = df_60percent [(df_60percent.index>np.percentile(df_60percent.index, 66)) & (df_60percent.index<=np.percentile(df_60percent.index, 100))].copy()


df_part3 = df_part3.drop( df_part3[ (df_part3['attack_cat']=='DoS') | (df_part3['attack_cat']=='Exploits') |  (df_part3['attack_cat']=='Normal')].index)


#target_names = ['Normal','Backdoor','Analysis','Fuzzers','Shellcode','Reconnaissance','Exploits','DoS','Worms','Generic']

df_part3['attack_cat'] = np.where( (df_part3['attack_cat']=='Analysis'),
                                   '1', np.where( (df_part3['attack_cat']=='Backdoor'),
                                        '2', np.where( (df_part3['attack_cat']=='Fuzzers'),
                                             '5', np.where( (df_part3['attack_cat']=='Generic'),
                                                '6', np.where( (df_part3['attack_cat']=='Reconnaissance'),
                                                     '7', np.where( (df_part3['attack_cat']=='Shellcode'),
                                                          '8', '9') ) ) ) ) )
#df_part3['attack_cat'] = np.where( (df_part2['attack_cat']=='DoS'), '3', '4')


df_part3.to_csv("/home/carlos/Projects/tfm/nb15/working/part3.csv",  sep=",", header= True, float_format='%.10g', index=False)

#################################################################

attacks_types = CategoricalDtype( categories = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms'], ordered=True)
df_40percent.attack_cat = df_40percent.attack_cat.astype(attacks_types).cat.codes



df_40percent.to_csv("/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_total_numeric_testing_WR_19FS-40percent.csv",  sep=",", header= True, float_format='%.10g', index=False)



df_60percent.attack_cat = df_60percent.attack_cat.astype(attacks_types).cat.codes



df_60percent.to_csv("/home/carlos/Projects/tfm/nb15/working/UNSW_NB15_total_numeric_testing_WR_19FS-60percent.csv",  sep=",", header= True, float_format='%.10g', index=False)

