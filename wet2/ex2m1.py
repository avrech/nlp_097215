import datetime
import os
import numpy as np
import sys
cur_file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_file_path)
os.chdir(cur_file_path)
from dependency_parser import DependencyParser

if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')
params = {
    'train_file': 'train.labeled',
    'train_sentences_max': None,  # set to None to use full set
    'test_sentences_max': None,  # set to None to use full set
    'test_file': 'test.labeled',
    'threshold': {4: 2, 8: 2, 10: 2},  # set thresholds for features appearance.
    # a feature that appears less than th times is filtered.
    'features_to_use': [1, 2, 3, 4, 5, 6, 8, 10, 13]
}

if True:
    # Choose None if to train a new model from scratch:
    model_file = None
else:
    # Choose path if to continue training some pre-trained model, for example:
    model_file = 'saved_models/2019-01-19/m1-m5000-test_acc-0.73-acc-0.75-from-10-52-31.pkl'

epochs = 100  # total num of epochs
snapshots = 10  # How many times to save model during training. if = 0 - do not train at all.
record_interval = 10  # evaluate model every num of epochs and store history for learning curve
eval_on = 100  # number of random samples to evaluate on.
shuffle = True  # shuffle training examples every epoch
model_description = 'm1'  # give a short description for model_name prefix
# At the finale of every training session,
# the model evaluates the entire train-set and test-set
# and reports results.
# you can plot history (learning curve at any time you want.
# all .pkl's and .png's files are saved.
# the save path is printed in console when save() occurs.

dp = DependencyParser(params, pre_trained_model_file=model_file)
dp.analyze_features()
for ii in range(snapshots):
    dp.train(epochs=int(np.ceil(epochs / snapshots)),
             record_interval=record_interval,
             eval_on=eval_on,
             shuffle=shuffle,
             model_description=model_description)
    dp.print_results()
# _, _, test_cm = dp.evaluate(dp.test_set, calc_confusion_matrix=True)
# dp.print_confusion_matrix(test_cm, print_to_csv=True, csv_id='test')
# _, _, train_cm = dp.evaluate(dp.train_set, calc_confusion_matrix=True)
# dp.print_confusion_matrix(train_cm, print_to_csv=True, csv_id='train')

dp.plot_history()
dp.model_info()
print('finished'.format(datetime.datetime.now()))
