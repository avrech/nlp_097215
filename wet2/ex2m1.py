import datetime
import time
from dependency_parser import read_anotated_file, DependencyParser, measure_accuracy

params = {
    'train_file': 'train.labeled',
    'train_sentences_max': None,
    'test_sentences_max': None,
    'test_file': 'test.labeled'
}

train_set = read_anotated_file(params['train_file'])[:params['train_sentences_max']]
test_set = read_anotated_file(params['test_file'])[:params['test_sentences_max']]

dp = DependencyParser(params)
print('Start training on {} samples...'.format(len(train_set)))
t_start = time.time()
dp.train(epochs=3, record_interval=1)

print('Evaluating test-set...')
test_acc, test_eval_time = dp.evaluate(test_set)

print('Evaluating train-set...')
train_acc, train_eval_time = dp.evaluate(train_set)




print('finished'.format(datetime.datetime.now()))
