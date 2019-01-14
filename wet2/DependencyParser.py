from tqdm import tqdm
import numpy as np

from wet2.chu_liu import Digraph


class DependencyParser:
    def __init__(self, params):
        self.true_graphs_dict = {}
        self.digraphs_dict = {}
        self.indexed_features = {}
        self.local_features_dict = {}
        self.global_features_dict = {}
        self.params = params
        self.num_of_features = 0
        self.param_vec = np.zeros(self.num_of_features)

    def train(self, sentences):
        # calc for each sentence: the true graph, a feature vector for each (h,m) in x, and the digraph
        for sentence in tqdm(sentences, 'Calculating train graphs...'):
            true_graph = self.calc_graph(sentence)
            self.true_graphs_dict[sentence] = true_graph

        features_dict = self.extract_features(sentences)
        curr_index = 0
        for k in features_dict:
            self.indexed_features[k] = {}
            for feature in features_dict[k]:
                self.indexed_features[k][feature] = curr_index
                curr_index += 1

        for sentence in tqdm(sentences, 'Calculating local and global features...'):
            self.local_features_dict[sentence] = self.calc_local_features(sentence)
            self.global_features_dict[sentence] = self.calc_global_features(sentence, self.true_graphs_dict[sentence])
            self.digraphs_dict[sentence] = self.prepare_digraph(sentence)

        # run Perceptron to find best parameter vector
        num_of_epochs = self.params['num_of_epochs']
        for n in tqdm(range(num_of_epochs), 'Running perceptron...'):
            for sentence in sentences:
                y_pred = self.digraphs_dict[sentence].mst()
                if y_pred != self.true_graphs_dict[sentence]:
                    new_w = self.param_vec + self.get_features_delta_vec(sentence, y_pred)
                    self.param_vec = new_w

    @staticmethod
    def calc_graph(sentence):
        """
        returns the true graph for an annotated sentence
        :param sentence: annotated sentence
        :return: {node_index: [index1, index2, ..], ..}
        """
        graph = {word[1]: [] for word in sentence}
        for word in sentence:
            graph[word[1]].append(word[0])
        return graph

    def extract_features(self, sentences):
        """
        getting the list of all features found in train set
        :param sentences:
        :return:
        """
        features_dict = {i: {} for i in range(1, 14)}
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
                    self.safe_add(features_dict[1], (p_word, p_pos))
                    self.safe_add(features_dict[2], p_word)
                    self.safe_add(features_dict[3], p_pos)
                    self.safe_add(features_dict[4], (c_word, c_pos))
                    self.safe_add(features_dict[5], c_word)
                    self.safe_add(features_dict[6], c_pos)
                    self.safe_add(features_dict[8], (p_pos, c_word, c_pos))
                    self.safe_add(features_dict[10], (p_word, p_pos, c_pos))
                    self.safe_add(features_dict[13], (p_pos, c_pos))
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
                local_features[(p_node, c_node)] = [x for x in local_features[(p_node, c_node)] if x is not None]
        return local_features

    def calc_global_features(self, sentence, y_graph):
        """
        local feature vectors are binary, but global are not necessarily so
        :param sentence:
        :param y_graph:
        :return: dict {index: count for positive index}
        """
        global_featurs = {}
        if sentence in self.local_features_dict:
            for source_node in y_graph:
                for target_node in y_graph[source_node]:
                    for positive_index in self.local_features_dict[sentence][(source_node, target_node)]:
                        if positive_index not in global_featurs:
                            global_featurs[positive_index] = 0
                        global_featurs[positive_index] += 1
        else:
            print('Error - no local features for sentence {}'.format(sentence))
        return global_featurs

    def infer(self, sentence):
        return None


def read_anotated_file(filename):
    f = open(filename)
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
    print(f'finished analyzing {filename} - found {len(sentences)} sentences, {len(words_list)} words '
          f'({len(words_set)} unique) and {len(pos_list)} parts of speech')
    print('POS list: ' + ','.join(pos_list))
    return sentences
