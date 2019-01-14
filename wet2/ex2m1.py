import datetime

from wet2.DependencyParser import read_anotated_file, DependencyParser

params = {
    'num_of_epochs': 10,
    'train_file': 'train.labeled',
    'test_file': 'test.labeled',
}

sentences = read_anotated_file(params['train_file'])
dp = DependencyParser(params)
dp.train(sentences)
print('{} - finished'.format(datetime.datetime.now()))
