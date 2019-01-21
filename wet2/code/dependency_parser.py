import pickle
from tqdm import tqdm
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import os
from chu_liu import Digraph
import csv


class DependencyParser:
    def __init__(self, params, pre_trained_model_file=None):
        """
        Initialize the parser.
        :param params: parameters for training.
        :param pre_trained_model_file: pickle file, contains the train_set, it's statistics dictionaries
                           and the corresponding pre-trained param_vec.
                           NOTE: if this parameter is not None, the params[train_file] is ignored,
                           and the test-set is loaded from the pickle,
                           in order to ensure compatibility of the param_vec and data.
        """
        # self.params = params
        # self.load_model = pre_trained_model
        self.test_set = read_file(params['test_file'])[:params['test_sentences_max']]
        self.last_train_time = 0
        self.model_name = None
        self.model_dir = None
        self.features_to_use = params['features_to_use']

        if pre_trained_model_file is not None:
            with open(pre_trained_model_file, 'rb') as f:
                model = pickle.load(f)
            self.threshold = model['threshold']
            self.true_graphs_dict = model['true_graphs_dict']
            self.indexed_features = model['indexed_features']
            self.num_of_features = model['num_of_features']
            self.param_vec = model['param_vec']
            self.train_set = model['train_set']
            self.history = model['history']
            self.total_train_time = model['total_train_time']
            self.init_epoch = model['final_epoch']
            self.results = model['results']
            model_path = pre_trained_model_file.split(sep='/')
            self.model_name = model_path[-1][:-4]
            self.model_dir = os.path.join(model_path[0], model_path[1])
        else:
            self.threshold = params['threshold']
            self.true_graphs_dict = {}
            self.indexed_features = {}
            self.num_of_features = 0
            self.param_vec = None
            self.train_set = read_file(params['train_file'])[:params['train_sentences_max']]
            self.history = []
            self.total_train_time = 0
            self.init_epoch = 0
            self.results = dict()
        self.local_features_dict = {}
        self.global_features_dict = {}
        self.digraphs_dict = {}
        self.confusion_mat = None

    def train(self, epochs=10, record_interval=0, eval_on=0, shuffle=True, model_description=''):
        """
        If load model is not None, the dictionaries assumed to be the same,
        and there is no need to calculate them again.
        Also, the param_vec get initial values from the pre-trained model,
        and we
        :return:
        """
        t_start = time.time()
        if self.true_graphs_dict.__len__() == 0:  # initialize dictionaries:
            # calc for each sentence: the true graph, a feature vector for each (h,m) in x, and the digraph
            for sentence in tqdm(self.train_set, 'Calculating train graphs...'):
                true_graph = self.calc_graph(sentence)
                self.true_graphs_dict[sentence] = true_graph

            features_dict = self.extract_features(self.train_set, threshold=self.threshold)
            curr_index = 0
            for k in features_dict:
                self.indexed_features[k] = {}
                for feature in features_dict[k]:
                    self.indexed_features[k][feature] = curr_index
                    curr_index += 1
            self.num_of_features = curr_index
            self.param_vec = np.zeros(self.num_of_features)

        for sentence in tqdm(self.train_set, 'Calculating local and global features...'):
            self.local_features_dict[sentence] = self.calc_local_features(sentence)
            self.global_features_dict[sentence] = self.calc_global_features(sentence, self.true_graphs_dict[sentence])
        for sentence in tqdm(self.train_set, 'Preparing Digraphs...'):
            self.digraphs_dict[sentence] = self.prepare_digraph(sentence)

        # run Perceptron to find best parameter vector
        # num_of_epochs = self.params['num_of_epochs']

        print('')
        print('----------------------------------')
        print('Running perceptron on train_set...')
        print('----------------------------------')

        for n in np.arange(self.init_epoch, self.init_epoch + epochs):
            if shuffle:
                shuffle_idx = np.random.permutation(self.train_set.__len__())
            else:
                shuffle_idx = np.arange(self.train_set.__len__())
            for sentence in tqdm([self.train_set[idx] for idx in shuffle_idx], 'Epoch no. {}'.format(n+1)):
                y_pred = self.digraphs_dict[sentence].mst().successors
                if y_pred != self.true_graphs_dict[sentence]:
                    new_w = self.param_vec + self.get_features_delta_vec(sentence, y_pred)
                    self.param_vec = new_w
            if record_interval > 0:
                if np.mod(n+1, record_interval) == 0:
                    trn_sel = np.random.permutation(self.train_set.__len__())
                    tst_sel = np.random.permutation(self.test_set.__len__())
                    train_acc, _, _ = self.evaluate([self.train_set[idx] for idx in trn_sel[:eval_on]])
                    test_acc, _, _ = self.evaluate([self.test_set[idx] for idx in tst_sel[:eval_on]])
                    self.history.append([n, np.mean(train_acc), test_acc])

        self.last_train_time = time.time() - t_start
        self.total_train_time += self.last_train_time
        final_epoch = self.init_epoch + epochs
        self.init_epoch += epochs

        # Evaluate train-set and test-set
        train_acc, train_eval_time, _ = self.evaluate(self.train_set)
        test_acc, test_eval_time, _ = self.evaluate(self.test_set)

        # update results
        self.results['Train-set size'] = len(self.train_set)
        self.results['Test-set size'] = len(self.test_set)
        self.results['# features'] = self.num_of_features
        self.results['# epochs'] = final_epoch
        self.results['Last Training Time [minutes]'] = "{:.2f}".format(self.last_train_time / 60)
        self.results['Total Training Time [minutes]'] = "{:.2f}".format(self.total_train_time / 60)
        self.results['Test-set accuracy'] = "{:.2f}".format(test_acc)
        self.results['Test-set evaluation time [minutes]'] = "{:.2f}".format(test_eval_time / 60)
        self.results['Train-set accuracy'] = "{:.2f}".format(train_acc)
        self.results['Train-set evaluation time [minutes]'] = "{:.2f}".format(train_eval_time / 60)

        # Save model to pkl:
        if model_description != '':
            model_description += '-'
        self.model_name = model_description + "s-{}-ep-{}-test_acc-{}-acc-{}-from-{}".format(
            len(self.train_set),
            final_epoch,
            self.results['Test-set accuracy'],
            self.results['Train-set accuracy'],
            str(datetime.datetime.now())[11:-7].replace(' ', '-').replace(':', '-'))

        model = dict()
        model['threshold'] = self.threshold
        model['true_graphs_dict'] = self.true_graphs_dict
        model['indexed_features'] = self.indexed_features
        model['num_of_features'] = self.num_of_features
        model['param_vec'] = self.param_vec
        model['train_set'] = self.train_set
        model['history'] = self.history
        model['total_train_time'] = self.total_train_time
        model['final_epoch'] = final_epoch
        model['results'] = self.results

        self.model_dir = os.path.join('saved_models', str(datetime.datetime.now())[:10].replace(' ', '-'))
        model_path = os.path.join(self.model_dir, self.model_name+".pkl")
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print('Save model to ' + model_path)
        return model_path

    def print_results(self):
        print('-------------------------------------------------')
        print('----------- Dependency-Parser Results -----------')
        print('-------------------------------------------------')
        print(tabulate(self.results.items(), headers=['Key', 'Value'], tablefmt='orgtbl', numalign='left'))

    def plot_history(self):
        if self.history.__len__() == 0:
            print('There is no history to plot')
            return
        fig = plt.figure(223)
        hist = np.array(self.history)
        plt.plot(hist[:, 0].astype(int), hist[:, 1:])
        plt.legend(['train acc', 'test acc'])
        plt.title('Learning Curve')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        fig_path = os.path.join(self.model_dir, self.model_name + "-learning_curve.png")
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        fig.savefig(fig_path)
        print('Save figure to ' + fig_path)

    @staticmethod
    def calc_graph(sentence):
        """
        returns the true graph for an annotated sentence
        :param sentence: annotated sentence - ((WORD1_ID, HEAD1_ID, POS1, DEP_LABEL1), (WORD2_ID, ...) ,... )
        :return: {node_index: [index1, index2, ..], ..}
                 The dict keys are the source nodes. and values are destination nodes,
                 representing directed edges from src_node to dst_node.
        """
        graph = {word[1]: [] for word in sentence}
        for word in sentence:
            graph[word[1]].append(word[0])
        return graph

    @staticmethod
    def get_discretize_sentence_len(sentence):
        if len(sentence) < 30:
            sentence_len = len(sentence)
        elif len(sentence) < 100:
            sentence_len = len(sentence) - (len(sentence) % 10)
        else:
            sentence_len = len(sentence) - (len(sentence) % 100)
        return sentence_len

    def extract_features(self, sentences, threshold=None):
        """
        getting the list of all features found in train set
        :param sentences:
        :param threshold:
        :return:
        """
        features_dict = {i: {} for i in range(1, 19)}
        distances = set()
        for sentence in tqdm(sentences, 'Extracting features...'):
            sentence_len = self.get_discretize_sentence_len(sentence)
            curr_graph = self.true_graphs_dict[sentence]
            for p_node in curr_graph:
                for c_node in curr_graph[p_node]:
                    if p_node == 0:
                        p_word = None
                        p_pos = None
                    else:
                        p_word = sentence[p_node-1][2]
                        p_pos = sentence[p_node-1][3]
                    if p_node < len(sentence):
                        p_next_pos = sentence[p_node][3]
                    else:
                        p_next_pos = None
                    c_next_pos = sentence[c_node][3] if c_node < len(sentence) else None
                    c_word = sentence[c_node-1][2]
                    c_pos = sentence[c_node-1][3]
                    distance = p_node - c_node
                    # for the basic model we use the feature set 1-13, except 7,9,11,12.
                    self.safe_add(features_dict[1], (p_word, p_pos))            # 1
                    self.safe_add(features_dict[2], p_word)                     # 2
                    self.safe_add(features_dict[3], p_pos)                      # 3
                    self.safe_add(features_dict[4], (c_word, c_pos))            # 4
                    self.safe_add(features_dict[5], c_word)                     # 5
                    self.safe_add(features_dict[6], c_pos)                      # 6
                    self.safe_add(features_dict[8], (p_pos, c_word, c_pos))     # 8
                    self.safe_add(features_dict[10], (p_word, p_pos, c_pos))    # 10
                    self.safe_add(features_dict[13], (p_pos, c_pos))            # 13
                    self.safe_add(features_dict[14], distance)                  # 14 distance
                    self.safe_add(features_dict[15], (p_pos, p_next_pos))       # 15 p_pos and next pos
                    self.safe_add(features_dict[16], (p_pos, c_pos, distance))  # 16 distance
                    self.safe_add(features_dict[17], sentence_len)              # 17 sentence length
                    if distance > 1:
                        self.safe_add(features_dict[18], (c_pos, c_next_pos, p_pos))# 18 helps discarding false far parents
                    distances.add(distance)
        # add features for all pos combinations (negative features)
        distances = [i - 20 for i in range(42)]
        pos_list = [pos for pos in features_dict[3]]
        for pos1 in pos_list:
            for pos2 in pos_list:
                self.safe_add(features_dict[13], (pos1, pos2))
                self.safe_add(features_dict[15], (pos1, pos2))
                for distance in distances:
                    self.safe_add(features_dict[16], (pos1, pos2, distance))

        # Thresholding:
        if threshold is not None:
            # Filter features that appear in dict:
            # a feature that appears less than th times is filtered
            for f, th in threshold.items():
                features_dict[f] = {k: v for k, v in features_dict[f].items() if v >= th}

        for feature_id in features_dict:
            if feature_id not in self.features_to_use:
                features_dict[feature_id] = {}

        return features_dict

    @staticmethod
    def safe_add(curr_dict, key):
        if key not in curr_dict:
            curr_dict[key] = 0
        curr_dict[key] += 1

    def prepare_digraph(self, sentence):
        """
        prepares a full graph on the sentence, with the appropriate scoring function
        :param sentence:
        :return: Digraph
        """
        graph_edges = self.get_full_graph(sentence)
        score_func = self.prepare_score_function(sentence)
        graph = Digraph(graph_edges, score_func)
        return graph

    def prepare_score_function(self, sentence):
        if sentence in self.local_features_dict:
            local_features = self.local_features_dict[sentence]
        else:
            local_features = self.calc_local_features(sentence)

        def sentence_graph_score(h, m):
            return self.get_dot_product(local_features[(h, m)], self.param_vec)

        return sentence_graph_score

    @staticmethod
    def get_dot_product(indices, v):
        dot_product = sum(v[indices])
        return dot_product

    @staticmethod
    def get_full_graph(sentence):
        all_indices = [x[0] for x in sentence]
        graph_edges = {i: [j for j in all_indices if j != i] for i in all_indices}
        graph_edges[0] = [i for i in all_indices]  # root
        return graph_edges

    def get_features_delta_vec(self, sentence, y_pred):
        true_features = self.global_features_dict[sentence]
        pred_features = self.calc_global_features(sentence, y_pred)
        features_key_set = set(true_features.keys()).union(set(pred_features.keys()))
        delta_vec = np.zeros(self.num_of_features)
        for feature_index in features_key_set:
            delta_vec[feature_index] = true_features.get(feature_index, 0) - pred_features.get(feature_index, 0)
        return delta_vec

    def calc_local_features(self, sentence):
        """
        calcaulates features for every arc (h, m) in the sentence
        :param sentence: all words are nodes
        :return: dictionary {(h, m): positive indices for h, m indices of words in sentence}
        """
        local_features = {}
        full_graph = self.get_full_graph(sentence)
        sentence_len = self.get_discretize_sentence_len(sentence)
        for p_node in full_graph:
            for c_node in full_graph[p_node]:
                if p_node == 0:
                    p_word = None
                    p_pos = None
                else:
                    p_word = sentence[p_node-1][2]
                    p_pos = sentence[p_node-1][3]
                if p_node < len(sentence):
                    p_next_pos = sentence[p_node][3]
                else:
                    p_next_pos = None
                c_next_pos = sentence[c_node][3] if c_node < len(sentence) else None
                c_word = sentence[c_node-1][2]
                c_pos = sentence[c_node-1][3]
                distance = p_node - c_node
                local_features[(p_node, c_node)] = [self.indexed_features[1].get((p_word, p_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[2].get(p_word, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[3].get(p_pos, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[4].get((c_word, c_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[5].get(c_word, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[6].get(c_pos, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[8].get((p_pos, c_word, c_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[10].get((p_word, p_pos, c_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[13].get((p_pos, c_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[14].get(distance, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[15].get((p_pos, p_next_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[16].get((p_pos, c_pos, distance), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[17].get(sentence_len, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[18].get((c_pos, c_next_pos, p_pos), None)]

                local_features[(p_node, c_node)] = [x for x in local_features[(p_node, c_node)] if x is not None]
        return local_features

    def calc_global_features(self, sentence, y_graph):
        """
        local feature vectors are binary, but global are not necessarily so
        :param sentence:
        :param y_graph:
        :return: dict {index: count for positive index}
        """
        global_features = {}
        if sentence in self.local_features_dict:
            for source_node in y_graph:
                for target_node in y_graph[source_node]:
                    for positive_index in self.local_features_dict[sentence][(source_node, target_node)]:
                        if positive_index not in global_features:
                            global_features[positive_index] = 0
                        global_features[positive_index] += 1
        else:
            print('Error - no local features for sentence {}'.format(sentence))
        return global_features

    def infer(self, sentence):
        """
        accepts sentence as a tuple, returns dependency graph
        :param sentence:
        :return:
        """
        digraph = self.prepare_digraph(sentence)
        return digraph.mst().successors

    def evaluate(self, sentences, verbose=False, calc_confusion_matrix=False):
        """
        Evaluate accuracy of the model as (# true_predictions / # words)
        :param sentences: annotated sentences.
        :param verbose: If True - print the infered and actual dependencies
        :param calc_confusion_matrix: If True - calculate confusion matrix
        :return: accuracy in [0,1], inference time.
        """
        t_start = time.time()
        """
        Calculate confusion matrix.
        format:
        confusion_mat[c_pos][true_p_pos][view_point][feature] = # of failures
        supported view_point's:
        ------------------------------------------------
        view_point | feature
        -----------|------------------------------------
        'distance' | child_index - parent_index (signed)
        'pred_pos' | the false predicted parent POS 
        """
        confusion_mat = None
        if calc_confusion_matrix:
            pos_set = set([w[3] for s in sentences for w in s])
            confusion_mat = {c_pos: {p_pos: {'distance': dict(),
                                             'pred_pos': dict(),
                                             'true_distance': dict()}
                                     for p_pos in pos_set}
                             for c_pos in pos_set}

        total_shot = []
        for sentence in tqdm(sentences, 'Evaluating model...'):
            true_successors = self.true_graphs_dict.get(sentence, self.calc_graph(sentence))
            pred_successors = self.infer(sentence)
            # calculate the head of each word in sentence
            true_deps = {c: p for p, children in true_successors.items() for c in children}
            pred_deps = {c: p for p, children in pred_successors.items() for c in children}
            shot = [true_deps[w] == pred_deps[w] for w in true_deps.keys()]
            total_shot += shot
            if verbose:
                print('infered: ' + str(pred_deps))
                print('actual:  ' + str(true_deps))

            if calc_confusion_matrix:
                """ for each failure, record the following statistics:
                    given c_pos and p_pos:
                    # failures as function of distance between c and p
                    # failures as function of the false predicted parent POS
                """
                for w_id in true_deps.keys():
                    if true_deps[w_id] != pred_deps[w_id]:
                        c = sentence[w_id-1]
                        true_p = sentence[true_deps[w_id]-1]
                        pred_p = sentence[pred_deps[w_id]-1]
                        # record the true distance between child and parent:
                        # self.safe_add(confusion_mat[c[3]][true_p[3]]['distance'], w_id - true_deps[w_id])

                        if w_id - true_deps[w_id] not in confusion_mat[c[3]][true_p[3]]['distance']:
                            confusion_mat[c[3]][true_p[3]]['distance'][w_id - true_deps[w_id]] = {'pred_pos': {}, 'pred_distance': {}}
                        # for the true distance, record the false predicted POS and distance:
                        self.safe_add(confusion_mat[c[3]][true_p[3]]['distance'][w_id - true_deps[w_id]]['pred_pos'], pred_p[3])
                        self.safe_add(confusion_mat[c[3]][true_p[3]]['distance'][w_id - true_deps[w_id]]['pred_distance'], w_id - pred_deps[w_id])

                        # record the false prediction POS:
                        self.safe_add(confusion_mat[c[3]][true_p[3]]['pred_pos'], pred_p[3])
                        # for each false predicted POS:

        acc = np.mean(total_shot)
        return np.mean(acc), time.time()-t_start, confusion_mat

    def print_confusion_matrix(self, cm, print_to_csv=True, save_pkl=True, csv_id='last_eval', print_to_console=False):
        """
        Print confusion matrix statistics
        :return: None
        """
        print('Computing ' + csv_id + ' confusion matrix...')
        # print failures vs. c_pos, p_pos
        headers = ['C\P'] + [c_pos for c_pos in cm.keys()] + ['Total']
        rows = []
        for child_pos, parents in cm.items():
            row = []
            for parent in parents.values():
                row.append(sum(parent['pred_pos'].values()))  # total_failures
            row.append(sum(row))
            rows.append([child_pos] + row)
        # calculate total failures per parent
        mat = np.array([[v for v in row[1:]] for row in rows])
        total_failures_per_parent = np.sum(mat, axis=0)
        rows.append(['Total:'] + list(total_failures_per_parent))
        if print_to_console:
            print('----------------------------------------------------------')
            print('----------- Dependency-Parser Confusion Matrix -----------')
            print('----------------------------------------------------------')
            print(tabulate(rows, headers=headers, tablefmt='orgtbl', numalign='left'))
        if save_pkl:
            csv_filename = os.path.join(self.model_dir, self.model_name + '-' + csv_id + '-cm.pkl')
            if not os.path.isdir(self.model_dir):
                os.mkdir(self.model_dir)
            with open(csv_filename, "wb") as f:
                pickle.dump(cm, f)
            print('Save confusion matrix to ' + csv_filename)

        if print_to_csv:
            csv_filename = os.path.join(self.model_dir, self.model_name + '-' + csv_id + '-confusion_mat.csv')
            with open(csv_filename, "w") as f:
                writer = csv.writer(f)
                writer.writerows([headers] + rows)

    def model_info(self):
        """
        print feature statistics according to homework guidelines.
        :return: None
        """
        # feature_statistics = {k: len(v) for k, v in dp.indexed_features.items()}
        feature_id = ['Feature ID'] + [str(k) for k in self.indexed_features.keys()]
        feature_cnt = ['Counts'] + [str(len(v)) for v in self.indexed_features.values()]
        print('----------------------------------------------')
        print('----------- Dependency-Parser Info -----------')
        print('----------------------------------------------')
        print(tabulate([feature_cnt], headers=feature_id, tablefmt='orgtbl', numalign='left'))

    def analyze_features(self):
        """
        This function extract features from the train-set and test-set,
        and analyze the statistics. It reports the overlap between the distributions.
        It can help to decide how to select or design features ad-hoc for maximizing
        the test accuracy.
        Features that do not appear in the test-set are probably useless.
        :return: None
        """
        # calc for each sentence:
        # the true graph,
        # a feature vector for each (h,m) in x,
        # and the digraph
        true_graph_dict_bak = self.true_graphs_dict
        self.true_graphs_dict = dict()
        for sentence in tqdm(self.train_set + self.test_set, 'Calculate true graphs for features analysis'):
            true_graph = self.calc_graph(sentence)
            self.true_graphs_dict[sentence] = true_graph
        th_trn_features = self.extract_features(self.train_set, threshold=self.threshold)
        full_trn_features = self.extract_features(self.train_set)
        tst_features = self.extract_features(self.test_set)
        th_trn_f_set = {k: set(f.keys()) for k, f in th_trn_features.items()}
        full_trn_f_set = {k: set(f.keys()) for k, f in full_trn_features.items()}
        diff_trn_f_set = {k: full_trn_f_set[k] - th_trn_f_set[k] for k in full_trn_f_set.keys()}
        tst_f_set = {k: set(f.keys()) for k, f in tst_features.items()}
        # The common features in train and test:
        th_f_set_intersection = {k: th_trn_f_set[k] & tst_f_set[k] for k in th_trn_f_set.keys()}
        diff_f_set_intersection = {k: diff_trn_f_set[k] & tst_f_set[k] for k in diff_trn_f_set.keys()}

        # The efficiency per feature type:
        th_f_overlap = {k: len(th_f_set_intersection[k])/len(th_trn_f_set[k]) for k in th_trn_f_set.keys() if len(th_trn_f_set[k]) > 0}
        # The damage because of thresholding: percents of test features which discarded.
        th_damage = {k: len(diff_f_set_intersection[k]) / len(tst_f_set[k]) for k in diff_trn_f_set.keys() if len(tst_f_set[k]) > 0}
        headers = ['Feature ID', 'Count (th on)', 'Overlap', 'Threshold Damage']
        statistics = np.array([[k, len(th_trn_f_set[k]), th_f_overlap.get(k, 0), th_damage.get(k, 0)] for k in th_trn_f_set.keys()])
        th_total_f = sum(statistics[:, 1])
        th_total_overlap = sum(statistics[:, 1] * statistics[:, 2])/th_total_f
        print('-----------------------------------------------------------')
        print('----------- Dependency-Parser Features Analysis -----------')
        print('-----------------------------------------------------------')
        print(tabulate(statistics, headers=headers, tablefmt='orgtbl', numalign='left'))
        print('Total Features in Train-Set (th on): ', int(th_total_f))
        print('Total Overlap: {:.2f}'.format(th_total_overlap))


        """
        Feature 8 is very sparse, and has 0.15 overlap
        see features sorted by occurance helps to define threshold."""

        from operator import itemgetter
        trn_f_sorted = dict()
        for f in th_trn_features.keys():
            trn_f_sorted[f] = {k: v for k, v in sorted(th_trn_features[f].items(), key=itemgetter(1), reverse=True)}

        # restore true graphs
        self.true_graphs_dict = dict()
        self.true_graphs_dict = true_graph_dict_bak


def read_file(filename, annotated=True):
    with open(filename, 'r') as f:
        sentences = []
        curr_sentence = []
        words_list = []
        pos_list = set()
        for row in f:
            if row.rstrip() == '':
                sentences.append(tuple(curr_sentence))
                curr_sentence = []
            else:
                split_row = row.rstrip().split('\t')
                if annotated:
                    curr_word = (int(split_row[0]), int(split_row[6]), split_row[1], split_row[3])
                else:
                    curr_word = (int(split_row[0]), None, split_row[1], split_row[3])
                curr_sentence.append(curr_word)
                words_list.append(curr_word[2])
                pos_list.add(curr_word[3])
        words_set = set(words_list)
        print('finished analyzing {} - found {} sentences, {} words '.format(filename, len(sentences), len(words_list)),
              '({} unique) and {} parts of speech'.format(len(words_set), len(pos_list), ))
        print('POS list: ' + ','.join(pos_list))
    return sentences


def annotate_file(filename, model, result_fname=None, result_dir='results'):
    if result_fname is None:
        print('Error in annotate_file. result_fname was not supported.')
        exit(-1)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    sentences = read_file(filename, annotated=False)
    new_file = os.path.join(result_dir, result_fname)
    with open(new_file, 'w') as f:
        for sentence in tqdm(sentences, 'annotating sentences'):
            pred_successors = model.infer(sentence)
            # calculate the head of each word in sentence
            pred_deps = {c: p for p, children in pred_successors.items() for c in children}
            for word in sentence:
                word_line = '\t'.join([str(word[0]), word[2], '_', word[3], '_', '_',
                                       str(pred_deps[word[0]]), '_', '_', '_'])
                f.write(word_line + '\n')
            f.write('\n')

