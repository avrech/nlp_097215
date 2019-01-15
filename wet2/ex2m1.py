import datetime

from wet2.DependencyParser import read_anotated_file, DependencyParser, measure_accuracy

params = {
    'num_of_epochs': 10,
    'train_file': 'train.labeled',
    'train_sentences_max': 50,
    'train_sentences_test_max': 10,
    'test_file': 'test.labeled',
}

sentences = read_anotated_file(params['train_file'])[:params['train_sentences_max']]
dp = DependencyParser(params)
dp.train(sentences)
measure_accuracy(dp, sentences[:params['train_sentences_test_max']])
print('{} - finished'.format(datetime.datetime.now()))
