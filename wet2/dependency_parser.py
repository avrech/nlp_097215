import pickle
from tqdm import tqdm
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import os

from wet2.chu_liu_py2 import Digraph


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
        self.test_set = read_anotated_file(params['test_file'])[:params['test_sentences_max']]
        self.last_train_time = 0
        self.model_name = None
        if pre_trained_model_file is not None:
            with open(pre_trained_model_file, 'rb') as f:
                model = pickle.load(f)
            self.threshold = model['threshold']
            self.true_graphs_dict = model['true_graphs_dict']
            self.indexed_features = model['indexed_features']
            self.local_features_dict = model['local_features_dict']
            self.global_features_dict = model['global_features_dict']
            self.num_of_features = model['num_of_features']
            self.param_vec = model['param_vec']
            self.train_set = model['train_set']
            self.history = model['history']
            self.total_train_time = model['total_train_time']
            self.init_epoch = model['final_epoch']
            self.results = model['results']
        else:
            self.threshold = params['threshold']
            self.true_graphs_dict = {}
            self.indexed_features = {}
            self.local_features_dict = {}
            self.global_features_dict = {}
            self.num_of_features = 0
            self.param_vec = None
            self.train_set = read_anotated_file(params['train_file'])[:params['train_sentences_max']]
            self.history = []
            self.total_train_time = 0
            self.init_epoch = 0
            self.results = dict()
        self.digraphs_dict = {}
        self.model_dir = None

    def train(self, epochs=10, record_interval=0, eval_on=0, shuffle=True, threshold=None):
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

            features_dict = self.extract_features(self.train_set, threshold=threshold)
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
                self.global_features_dict[sentence] = \
                    self.calc_global_features(sentence, self.true_graphs_dict[sentence])
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
                if np.mod(n, record_interval) == 0:
                    trn_sel = np.random.permutation(self.train_set.__len__())
                    tst_sel = np.random.permutation(self.test_set.__len__())
                    train_acc, _ = self.evaluate([self.train_set[idx] for idx in trn_sel[:eval_on]])
                    test_acc, _ = self.evaluate([self.test_set[idx] for idx in tst_sel[:eval_on]])
                    self.history.append([n, np.mean(train_acc), test_acc])

        self.last_train_time = time.time() - t_start
        self.total_train_time += self.last_train_time
        final_epoch = self.init_epoch + epochs
        self.init_epoch += epochs

        # Evaluate train-set and test-set
        train_acc, train_eval_time = self.evaluate(self.train_set)
        test_acc, test_eval_time = self.evaluate(self.test_set)

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
        self.model_name = "m{}-test_acc-{}-acc-{}-from-{}".format(
            len(self.train_set),
            self.results['Test-set accuracy'],
            self.results['Train-set accuracy'],
            str(datetime.datetime.now())[11:-7].replace(' ', '-').replace(':', '-'))

        model = dict()
        model['threshold'] = self.threshold
        model['true_graphs_dict'] = self.true_graphs_dict
        model['indexed_features'] = self.indexed_features
        model['local_features_dict'] = self.local_features_dict
        model['global_features_dict'] = self.global_features_dict
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

    def extract_features(self, sentences, threshold=None):
        """
        getting the list of all features found in train set
        :param sentences:
        :param threshold:
        :return:
        """
        features_dict = {i: {} for i in range(1, 15)}
        for sentence in tqdm(sentences, 'Extracting features...'):
            # TODO: get features for the full graph?
            curr_graph = self.true_graphs_dict[sentence]
            for p_node in curr_graph:
                for c_node in curr_graph[p_node]:
                    if p_node == 0:
                        p_word = None
                        p_pos = None
                    else:
                        p_word = sentence[p_node-1][2]
                        p_pos = sentence[p_node-1][3]
                    c_word = sentence[c_node-1][2]
                    c_pos = sentence[c_node-1][3]
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
                    self.safe_add(features_dict[14], (p_node - c_node))         # distance
        # Thresholding:
        if threshold is not None:
            # Filter features that appear in dict:
            # a feature that appears less than th times is filtered
            for f, th in threshold.items():
                features_dict[f] = {k: v for k, v in features_dict[f].items() if v >= th}

        return features_dict

    @staticmethod
    def safe_add(curr_dict, key):
        if key not in curr_dict:
            curr_dict[key] = 0
        curr_dict[key] += 1

    def prepare_digraph(self, sentence):  # TODO: verify functionality on non-annotated sentences.
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

        # TODO: make sure this works...
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
        for p_node in full_graph:
            for c_node in full_graph[p_node]:
                if p_node == 0:
                    p_word = None
                    p_pos = None
                else:
                    p_word = sentence[p_node-1][2]
                    p_pos = sentence[p_node-1][3]
                c_word = sentence[c_node-1][2]
                c_pos = sentence[c_node-1][3]
                local_features[(p_node, c_node)] = [self.indexed_features[1].get((p_word, p_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[2].get(p_word, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[3].get(p_pos, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[4].get((c_word, c_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[5].get(c_word, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[6].get(c_pos, None)]
                local_features[(p_node, c_node)] += [self.indexed_features[8].get((p_pos, c_word, c_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[10].get((p_word, p_pos, c_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[13].get((p_pos, c_pos), None)]
                local_features[(p_node, c_node)] += [self.indexed_features[14].get((p_node - c_node), None)]

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

    def evaluate(self, sentences, verbose=False):
        """
        Evaluate accuracy of the model as (# true_predictions / # words)
        :param sentences: annotated sentences.
        :param verbose:
        :return: accuracy in [0,1], inference time.
        """
        t_start = time.time()
        total_shot = []
        for sentence in sentences:
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

        acc = np.mean(total_shot)
        return np.mean(acc), time.time()-t_start

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
        trn_features = self.extract_features(self.train_set, threshold=self.threshold)
        tst_features = self.extract_features(self.test_set)
        trn_f_set = {k: set(f.keys()) for k, f in trn_features.items()}
        tst_f_set = {k: set(f.keys()) for k, f in tst_features.items()}
        # The common features in train and test:
        f_set_intersection = {k: trn_f_set[k] & tst_f_set[k] for k in trn_f_set.keys()}
        # The efficiency per feature type:
        f_overlap = {k: len(f_set_intersection[k])/len(trn_f_set[k]) for k in trn_f_set.keys() if len(trn_f_set[k]) > 0}
        headers = ['Feature ID', 'Count', 'Overlap']
        statistics = np.array([[k, len(trn_f_set[k]), f_overlap.get(k, 0)] for k in trn_f_set.keys()])
        total_f = sum(statistics[:, 1])
        total_overlap = sum(statistics[:, 1] * statistics[:, 2])/total_f
        print('-----------------------------------------------------------')
        print('----------- Dependency-Parser Features Analysis -----------')
        print('-----------------------------------------------------------')
        print(tabulate(statistics, headers=headers, tablefmt='orgtbl', numalign='left'))
        print('Total Features in Train-Set: ', int(total_f))
        print('Total Overlap: {:.2f}'.format(total_overlap))

        """
        Feature 8 is very sparse, and has 0.15 overlap
        see features sorted by occurance helps to define threshold."""

        from operator import itemgetter
        trn_f_sorted = dict()
        for f in trn_features.keys():
            trn_f_sorted[f] = {k: v for k, v in sorted(trn_features[f].items(), key=itemgetter(1), reverse=True)}

        # restore true graphs
        self.true_graphs_dict = dict()
        self.true_graphs_dict = true_graph_dict_bak


def read_anotated_file(filename):
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
                curr_word = (int(split_row[0]), int(split_row[6]), split_row[1], split_row[3])
                curr_sentence.append(curr_word)
                words_list.append(curr_word[2])
                pos_list.add(curr_word[3])
        words_set = set(words_list)
        print('finished analyzing {} - found {} sentences, {} words '.format(filename, len(sentences), len(words_list)),
              '({} unique) and {} parts of speech'.format(len(words_set), len(pos_list), ))
        print('POS list: ' + ','.join(pos_list))
    return sentences


def measure_accuracy(dp, sentences):
    for sentence in sentences:
        infered_dp = dp.infer(sentence)
        print('infered: ' + str(infered_dp))
        print('actual:  ' + str(dp.true_graphs_dict[sentence]))

if __name__ == '__main__':
    pass
