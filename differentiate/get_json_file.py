import pandas as pd
import os

__file__ = 'C:/Users/jtros/CS/cours/PoleProjet/FormationRecherche/Tsunami/TP/sceance4/Tsunami'

print('\n cwd:', os.getcwd())
os.chdir(__file__)
print('changed to:', os.getcwd(), '\n')


df = pd.read_json(
    'differentiate/my_dir/tune_hypermodel/trial_0/trial.json')
for col in df.columns:
    print(col)

print(df['score'])

print("DataFrame generated using JSON file:")
print(df)
