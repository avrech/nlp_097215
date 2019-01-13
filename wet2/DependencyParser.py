from tqdm import tqdm
import numpy as np

from wet2.chu_liu import Digraph


class DependencyParser:
    def __init__(self, params):
        self.true_graphs_dict = {}
        self.pred_graphs_dict = {}
        self.features_dict = {}
        self.params = params
        self.num_of_features = 0
        self.param_vec = np.zeros(self.num_of_features)

    def train(self, sentences):
        # calc graph y for each sentence x
        for sentence in tqdm(sentences, 'Calculating train graphs...'):
            self.true_graphs_dict[sentence] = self.calc_graph(sentence)
            self.pred_graphs_dict[sentence] = self.prepare_digraph(sentence)
        # calc feature vector for each (x,y) pair
        for sentence in tqdm(sentences, 'Calculating vector features...'):
            self.features_dict[sentence] = self.calc_global_features(sentence, self.true_graphs_dict[sentence])

        # run Perceptron to find best parameter vector
        num_of_epochs = self.params['num_of_epochs']
        for n in tqdm(range(num_of_epochs), 'Running perceptron...'):
            for sentence in sentences:
                y_pred = self.pred_graph_dict[sentence].mst()
                if y_pred != self.true_graphs_dict[sentence]:
                    new_w = self.param_vec + self.features_dict[sentence] - self.calc_global_features(sentence, y_pred)
                    self.param_vec = new_w

    def calc_graph(self, sentence):
        # Digraph(graph, self.get_graph_score)
        return None

    def calc_global_features(self, sentence, graph):
        return None

    def get_graph_score(self, h, m):
        # uses self.param_vec to calculate score
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
