import datetime
import os
import numpy as np

from dependency_parser import DependencyParser

os.chdir(os.getcwd())
if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')
params = {
    'train_file': 'train.labeled',
    'train_sentences_max': 200,  # set to None to use full set
    'test_sentences_max': 50,  # set to None to use full set
    'test_file': 'test.labeled',
    'threshold': {4: 2, 8: 2, 10: 2}  # set thresholds for features appearance.
    # a feature that appears less than th times is filtered.
}

if True:
    # Choose None if to train a new model from scratch:
    model_file = None
else:
    # Choose path if to continue training some pre-trained model, for example:
    model_file = 'saved_models/2019-01-17/m100-test_acc-0.31-acc-0.72-from-00:31:10.pkl'

epochs = 20  # total num of epochs
snapshots = 10  # How many times to save model during training
record_interval = 5  # evaluate model every num of epochs and store history for learning curve
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

dp.plot_history()
dp.model_info()
print('finished'.format(datetime.datetime.now()))
