import datetime

from wet1.MEMM import evaluate, MEMM, get_parsed_sentences_from_tagged_file, tag_file

if __name__ == "__main__":
    # To run a model from scrach: set from_pickle: False, enter train and test file names, max num of sentences for
    # training (train_set_size) and maximal number of iterations (max_iter)
    params = {
        'train_file': 'train.wtag',
        'train_set_size': 5000,
        'from_pickle': False,
        'pickle_input': 'model_prm - 5000 train 10 percent affix 110iter.pkl',
        'pickle_output': 'model_prm.pkl',
        'maxiter': 110,
        'test_file': 'test.wtag',
        'test_train_size': 5000,
        'test_set_size': 1000,
        'affix_precent': 90,
        'beam_min': 10,
        'competition_file': 'comp.words',
    }

    my_model = MEMM(params)
    parsed_sentences = get_parsed_sentences_from_tagged_file(params['train_file'])
    my_model.train_model(parsed_sentences[:params['train_set_size']])

    # ecvaluate train set
    print('====================== testing accuracy on train data ======================')
    train_infer_start = datetime.datetime.now()
    train_acc = evaluate(my_model, params['train_file'], params['test_train_size'])
    train_infer_end = datetime.datetime.now()

    # Evaluate test set:
    print('====================== testing accuracy on test data ======================')
    test_infer_start = datetime.datetime.now()
    test_acc = evaluate(my_model, params['test_file'], params['test_set_size'])
    test_infer_end = datetime.datetime.now()

    # tagging competition
    print('====================== testing accuracy on train data ======================')
    train_infer_start = datetime.datetime.now()
    train_acc = tag_file(my_model, params['competition_file'])
    train_infer_end = datetime.datetime.now()

    print('train size,start time,finish time,total time,max iter,last L(v),train test size,train acc,train infer start,'
          'train infer end,train infer total,test infer start,test infer end,test infer total,test set size,test acc')
    print(f'{params["train_set_size"]},'
          f'{my_model.train_start},'
          f'{my_model.train_end},'
          f'{my_model.train_end - my_model.train_start},'
          f'{params["maxiter"]},'
          f'{my_model.last_known_v},'
          f'{params["test_train_size"]},'
          f'{train_acc},'
          f'{train_infer_start},'
          f'{train_infer_end},'
          f'{train_infer_end - train_infer_start},'
          f'{test_infer_start},'
          f'{test_infer_end},'
          f'{train_infer_end - train_infer_start},'
          f'{params["test_set_size"]},'
          f'{test_acc}')
