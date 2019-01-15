import datetime
import time
from dependency_parser import read_anotated_file, DependencyParser, measure_accuracy
from tabulate import tabulate
params = {
    'num_of_epochs': 100,
    'train_file': 'train.labeled',
    'train_sentences_max': None,
    'test_sentences_max': None,
    'test_file': 'test.labeled'
}

train_set = read_anotated_file(params['train_file'])[:params['train_sentences_max']]
test_set = read_anotated_file(params['test_file'])[:params['test_sentences_max']]
results = dict()
results['Train-set size'] = len(train_set)
results['Test-set size'] = len(test_set)

dp = DependencyParser(params)
print('Start training on {} samples...'.format(len(train_set)))
t_start = time.time()
dp.train(train_set)
results['# features'] = dp.num_of_features
results['# epochs'] = params['num_of_epochs']
results['Training time [minutes]'] = "{:.2f}".format(time.time() - t_start/60)

print('Evaluating test-set...')
t_start = time.time()
results['Test-set accuracy'] = "{:.2f}".format(dp.evaluate(test_set))
results['Test-set evaluation time [minutes]'] = "{:.2f}".format(time.time() - t_start/60)

print('Evaluating train-set...')
t_start = time.time()
results['Train-set accuracy'] = "{:.2f}".format(dp.evaluate(train_set))
results['Train-set evaluation time [minutes]'] = "{:.2f}".format(time.time() - t_start/60)



print('-------------------------------')
print('----------- Results -----------')
print('-------------------------------')
print(tabulate(results.items(), headers=['Param', 'Val'], tablefmt='orgtbl', numalign='left'))

print('finished'.format(datetime.datetime.now()))
