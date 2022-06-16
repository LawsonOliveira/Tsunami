from MyTunerRandomSearch import path_to_trials
import os
import numpy as np
import pandas as pd
import json
import keras
import matplotlib.pyplot as plt

__file__ = 'C:/Users/jtros/CS/cours/PoleProjet/FormationRecherche/Tsunami/TP/sceance4/Tsunami'

print('\ncwd:\n', os.getcwd())
os.chdir(__file__)
print('changed to:\n', os.getcwd(), '\n')


# get best trial
best_score = +np.inf
best_hyperparameters = dict()
path_to_best_trial = ''
for trial_folder in os.listdir(path_to_trials):
    if trial_folder[:5] == 'trial':
        # print(trial_folder)
        path_to_trial = path_to_trials+'/'+trial_folder
        df = pd.read_json(
            path_to_trial+'/trial.json')
        score = df['score'][0]
        if best_score > score:
            best_score = score
            best_hyperparameters = df['hyperparameters']['values']
            path_to_best_trial = path_to_trial


def get_best_trial(do_load_model=False):
    '''
    remark: MyTunerRandomSearch must have ended at least one trial
    '''
    print('best score:\n', best_score)
    print('best hyperparameters:\n', best_hyperparameters)

    with open(path_to_best_trial+'/history.json', 'r') as fp:
        best_history = json.load(fp)
    plt.plot(np.arange(len(best_history['val_mae'])),
             best_history['val_mae'], label='val_mae')
    plt.plot(np.arange(len(best_history['train_loss'])),
             best_history['train_loss'], label='train_loss')
    plt.title('Best model history')
    plt.legend()
    plt.show()

    if do_load_model:
        model = keras.models.load_model(path_to_trial+'/checkpoint.h5')
        return model
    return None


if __name__ == '__main__':
    get_best_trial()
