import datetime
import os
import numpy as np
import sys
sys.path.append(os.getcwd())
os.chdir(os.getcwd())
from dependency_parser import DependencyParser

if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')
params = {
    'train_file': 'train.labeled',
    'train_sentences_max': None,  # set to None to use full set
    'test_sentences_max': None,  # set to None to use full set
    'test_file': 'test.labeled',
    'threshold': {4: 2, 8: 2, 10: 2}  # set thresholds for features appearance.
    # a feature that appears less than th times is filtered.
}

if True:
    # Choose None if to train a new model from scratch:
    model_file = None
else:
    # Choose path if to continue training some pre-trained model, for example:
    model_file = "saved_models/2019-01-17/m5000-test_acc-0.25-acc-0.29-from-22:50:17.pkl"

epochs = 1  # total num of epochs
snapshots = 1  # How many times to save model during training. if = 0 - do not train at all.
record_interval = 0  # evaluate model every num of epochs and store history for learning curve
eval_on = 20  # number of random samples to evaluate on.
shuffle = True  # shuffle training examples every epoch
# At the finale of every training session,
# the model evaluates the entire train-set and test-set
# and reports results.
# you can plot history (learning curve at any time you want.
# all .pkl's and .png's files are saved.
# the save path is printed in console when save() occurs.

dp = DependencyParser(params, pre_trained_model_file=model_file)
dp.analyze_features()
for ii in range(snapshots):
    dp.train(epochs=np.ceil(epochs / snapshots),
             record_interval=record_interval,
             eval_on=eval_on,
             shuffle=shuffle)
    dp.print_results()
_, _, test_confusion_mat = dp.evaluate(dp.test_set, calc_confusion_matrix=True)
dp.plot_history()
dp.model_info()
print('finished'.format(datetime.datetime.now()))
