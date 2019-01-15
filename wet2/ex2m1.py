import datetime
import time
from dependency_parser import read_anotated_file, DependencyParser, measure_accuracy
from tabulate import tabulate
params = {
    'num_of_epochs': 50,
    'train_file': 'train.labeled',
    'train_sentences_max': 200,
    'test_sentences_max': 100,
    'test_file': 'test.labeled'
}

train_set = read_anotated_file(params['train_file'])[:params['train_sentences_max']]
test_set = read_anotated_file(params['test_file'])[:params['test_sentences_max']]
dp = DependencyParser(params)
t_start = time.time()
dp.train(train_set)
training_time = time.time() - t_start
test_acc = dp.evaluate(test_set)
train_acc = dp.evaluate(train_set)
results = {}
results['Train-set size'] = params['train_sentences_max']
results['Test-set size'] = params['test_sentences_max']
results['# features'] = dp.num_of_features
results['# epochs'] = params['num_of_epochs']
results['Training time [sec]'] = "{:.2f}".format(training_time)
results['Train-set accuracy'] = "{:.2f}".format(train_acc)
results['Test-set accuracy'] = "{:.2f}".format(test_acc)
print('-------------------------------')
print('----------- Results -----------')
print('-------------------------------')
print(tabulate(results.items(), headers=['Param', 'Val'], tablefmt='orgtbl', numalign='left'))

print('finished'.format(datetime.datetime.now()))
