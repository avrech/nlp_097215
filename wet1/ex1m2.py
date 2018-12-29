import datetime

from wet1.MEMM import evaluate, MEMM, get_parsed_sentences_from_tagged_file, tag_file, compare_files

if __name__ == "__main__":
    # load training set
    params = {
        'train_file': 'train2.wtag',
        'train_set_size': 5000,
        'from_pickle': True,
        'pickle_input': 'model_prm - model2 60iter 50affix.pkl',
        'pickle_output': 'model_prm.pkl',
        'maxiter': 0,
        'test_file': 'test.wtag',
        'test_train_size': 0,
        'test_set_size': 0,
        'affix_precent': 50,
        'beam_min': 10,
        'competition_file': 'comp2.words',
    }

    parsed_sentences = get_parsed_sentences_from_tagged_file(params['train_file'])
    my_model = MEMM(params)
    my_model.train_model(parsed_sentences[:params['train_set_size']])

    compare_files(params['train_file'], 'train2_output.txt', my_model.tags)

    # ecvaluate train set
    print('====================== testing accuracy on train data ======================')
    train_infer_start = datetime.datetime.now()
    train_acc = evaluate(my_model, params['train_file'], params['test_train_size'])
    train_infer_end = datetime.datetime.now()

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
          f'{params["test_set_size"]},'
          )
