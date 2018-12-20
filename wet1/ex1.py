# from collections import Counter
import numpy as np
import math
import datetime
import scipy
from scipy import optimize
import pickle
import time


class Context:
    def __init__(self, word, tag, history, prev_tag, prev_prev_tag, next_word):
        self.word = word
        self.tag = tag
        self.history = history
        self.prev_tag = prev_tag
        self.prev_prev_tag = prev_prev_tag
        self.next_word = next_word

    @staticmethod
    def get_context_tagged(sentence, index):
        """Creates a context from a tagged sentence"""
        word = sentence[index][0]
        tag = sentence[index][1]
        # TODO: add start / stop signs to history / next word?
        history = [w for w in sentence[:index]]
        prev_tag = sentence[index - 1][1] if index > 0 else None
        prev_prev_tag = sentence[index - 2][1] if index > 1 else None
        next_word = sentence[index + 1][0] if len(sentence) > index + 1 else None
        return Context(word, tag, history, prev_tag, prev_prev_tag, next_word)

    @staticmethod
    def get_context_untagged(sentence, index, tag, prev_prev_tag, prev_tag):
        """Creates a context from an untagged sentence"""
        word = sentence[index]
        # TODO: add start / stop signs to history / next word?
        history = [w for w in sentence[:index]]
        next_word = sentence[index + 1] if len(sentence) > index + 1 else None
        return Context(word, tag, history, prev_tag, prev_prev_tag, next_word)


class MEMM:
    def __init__(self):
        self.tags = []
        self.enriched_tags = []
        self.feature_set = []
        self.feature_set1 = []
        self.word_vectors = None
        self.sentences = None
        self.parameter_vector = None
        self.empirical_counts = None

    def train_model(self, sentences, param_vec=None):
        # prepare data and get statistics
        print(f'{datetime.datetime.now()} - processing data')
        word_tag_pairs = []
        for sentence in sentences:
            for word_tag in sentence:
                if word_tag not in word_tag_pairs:
                    word_tag_pairs.append(word_tag)
                if word_tag[1] not in self.tags:
                    self.tags.append(word_tag[1])

        self.enriched_tags = [None] + self.tags

        # define the features set
        # feature-word pairs in dataset
        self.feature_set += [(lambda w, t:(lambda cntx: 1 if cntx.word == w and cntx.tag == t else 0))(w, t)
                             for w, t in word_tag_pairs]
        # prefixes <= 4 and tag pairs in dataset
        # suffixes <= 4 and tag pairs in dataset
        # tag trigrams in datset
        # tag bigrams in datset
        # tag unigrams in datset
        # previous word + current tag pairs
        # next word + current tag pairs

        # for each word in the corpus, find its feature vector
        word_vectors = []
        for sentence in sentences:
            for i in range(len(sentence)):
                context = Context.get_context_tagged(sentence, i)
                vector = self.get_feature_vector_for_context(context)
                word_vectors.append(vector)
        self.word_vectors = np.array(word_vectors)
        self.sentences = sentences
        self.empirical_counts = np.sum(self.word_vectors, axis=0)

        # calculate the parameters vector
        print(f'{datetime.datetime.now()} - finding parameter vector')
        if param_vec is None:
            param_vec = scipy.optimize.minimize(fun=self.l, x0=np.ones(len(self.feature_set)), method='L-BFGS-B',
                                                jac=self.grad_l)
        # param_vec = scipy.optimize.fmin_l_bfgs_b(self.l, np.zeros(len(self.feature_set)), self.grad_l)
        # optimize.minimize(self.l,np.zeros(len(self.feature_set)),)
        self.parameter_vector = param_vec.x
        print(self.parameter_vector)
        print(f'{datetime.datetime.now()} - model train complete')

    # use Viterbi to infer tags for the target sentence
    def infer(self, sentence):
        t_start = time.time()
        parsed_sentence = [word.rstrip() for word in sentence.split(' ')]
        pi = {(0, None, None): 1}
        bp = {}
        for word_index in range(1, len(parsed_sentence) + 1):
            norm_for_context = {}
            pi_temp = {}
            bp_temp = {}
            for tag in self.tags:
                for prev_tag in self.enriched_tags:

                    past_proba = {prev_prev_tag: pi.get((word_index - 1, prev_prev_tag, prev_tag), 0) for prev_prev_tag
                                  in self.enriched_tags}
                    for prev_prev_tag in self.enriched_tags:
                        if past_proba[prev_prev_tag] != 0:
                            if (word_index, prev_tag, prev_prev_tag) not in norm_for_context:
                                norm_for_context[(word_index, prev_tag, prev_prev_tag)] = \
                                    self.get_context_norm(Context.get_context_untagged(parsed_sentence, word_index - 1,
                                                                                       tag, prev_tag, prev_prev_tag))
                    transition_proba = \
                        {prev_prev_tag: self.get_tag_proba(tag, Context.get_context_untagged(parsed_sentence,
                                                                                             word_index - 1, tag,
                                                                                             prev_tag, prev_prev_tag),
                                                           norm=norm_for_context[(word_index, prev_tag, prev_prev_tag)])
                         for prev_prev_tag in self.enriched_tags if past_proba[prev_prev_tag] != 0}
                    pi_temp[(word_index, prev_tag, tag)] = max([past_proba.get(prev_prev_tag, 0) *
                                                               transition_proba.get(prev_prev_tag, 0)
                                                               for prev_prev_tag in self.enriched_tags])
                    bp_temp[(word_index, prev_tag, tag)] = np.argmax(
                        [past_proba.get(prev_prev_tag, 0) * transition_proba.get(prev_prev_tag, 0) for prev_prev_tag in
                         self.enriched_tags])
            # trim entries with low probability (beam search)
            # find the probabilty above which we have at least BEAM_MIN possibilities
            min_proba_for_stage = 1
            count_entries_per_proba = len([pi_val for pi_key, pi_val in pi_temp.items()
                                           if pi_val >= min_proba_for_stage])
            BEAM_MIN = 10
            while count_entries_per_proba < BEAM_MIN:
                min_proba_for_stage = max([pi_val for pi_key, pi_val in pi_temp.items()
                                           if pi_val < min_proba_for_stage])
                count_entries_per_proba = len([pi_val for pi_key, pi_val in pi_temp.items()
                                               if pi_val >= min_proba_for_stage])
            # merge all the entries above min_proba_for_stage into pi/bp
            for key, val in [(key, val) for key, val in pi_temp.items() if val >= min_proba_for_stage]:
                pi[key] = val
                bp[key] = bp_temp[key]

        # use backpointers to find the tags
        sorted_pi = sorted([(k, u, v, pi[(k, u, v)]) for k, u, v in pi],  key=lambda x: x[3], reverse=True)
        index, tn_prev, tn, proba = [x for x in sorted_pi if x[0] == len(parsed_sentence)][0]
        result_tags = [tn_prev, tn]
        for word_index in range(len(parsed_sentence) - 1, 1, -1):
            index, tn_prev, tn, proba = [x for x in sorted_pi if x[0] == word_index and x[2] == result_tags[0]][0]
            result_tags = [tn_prev] + result_tags
        infer_time = time.time() - t_start
        return result_tags, infer_time

    def get_feature_vector_for_context(self, context):
        vector = [feature(context) for feature in self.feature_set]
        return vector

    def get_dot_product(self, feature_vector):
        dot_product = sum([feature_vector[i] * self.parameter_vector[i] for i in range(len(feature_vector))])
        return dot_product

    # soft max
    def get_tag_proba(self, tag, context, norm=None):
        context.tag = tag
        tag_vector = self.get_feature_vector_for_context(context)
        numerator = 2 ** self.get_dot_product(tag_vector)
        if norm is None:
            norm = self.get_context_norm(context)
        proba = numerator / norm if norm > 0 else 0
        return proba

    def get_context_norm(self, context):
        norm = 0
        for curr_tag in self.tags:
            context.tag = curr_tag
            tag_vector = self.get_feature_vector_for_context(context)
            norm += 2 ** self.get_dot_product(tag_vector)
        return norm

    # the ML estimate maximization function
    def l(self, v):
        # proba part
        proba = sum(np.dot(self.word_vectors, v))
        # normalization part
        norm_part = 0
        for sentence in self.sentences:
            for i in range(len(sentence)):
                curr_context = Context.get_context_tagged(sentence, i)
                curr_exp = 0
                for tag in self.tags:
                    curr_context.tag = tag
                    vector = self.get_feature_vector_for_context(curr_context)
                    curr_exp += 2 ** np.dot(v, vector)
                norm_part += math.log(curr_exp, 2)
        return -(proba - norm_part)

    def grad_l(self, v):
        # expected counts
        expected_counts = 0
        for sentence in self.sentences:
            for i in range(len(sentence)):
                curr_context = Context.get_context_tagged(sentence, i)
                curr_exp = 0
                normalization = 0
                nominator = 0
                for tag in self.tags:
                    curr_context.tag = tag
                    vector = self.get_feature_vector_for_context(curr_context)
                    curr_exp = 2 ** np.dot(v, vector)
                    normalization += curr_exp
                    nominator += np.dot(vector, curr_exp)

                expected_counts += nominator / normalization if normalization > 0 else 0
        return -(self.empirical_counts - expected_counts)

    def test(self, text, annotated=False):
        pass


# receives input file (of tagged sentences),
# returns a list of sentences, each sentence is a list of (word,tag)s
# also performs base verification of input
def get_parsed_sentences_from_tagged_file(filename):
    print(f'{datetime.datetime.now()} - loading data from file {filename}')
    f_text = open(filename)
    all_words_and_tags = []
    sentences = []
    for row in f_text:
        tagged_words = [word.rstrip() for word in row.split(' ')]
        words_and_tags = [tagged_word.split('_') for tagged_word in tagged_words]
        sentences.append(words_and_tags)
        all_words_and_tags += words_and_tags
    bad_words = [word for word in all_words_and_tags if len(word) != 2]
    if bad_words:
        print(f'found {len(bad_words)} bad words - {bad_words}')
    all_tags_list = [word[1] for word in all_words_and_tags]
    all_tags_set = set(all_tags_list)
    # tags_counter = Counter(all_tags_list)
    # all_words_list = [word[0] for word in all_words_and_tags]
    # words_counter = Counter(all_words_list)
    # all_words_set = set(all_words_list)
    print(f'{datetime.datetime.now()} - found {len(all_tags_set)} tags - {all_tags_set}')
    return sentences


if __name__ == "__main__":
    # load training set
    parsed_sentences = get_parsed_sentences_from_tagged_file('train.wtag')
    model = MEMM()
    model.train_model(parsed_sentences[:10])
    with open('model_prm.pkl', 'wb') as f:
        pickle.dump(model.parameter_vector, f)
    f = open('model_prm.pkl', 'rb')
    param_vector = pickle.load(f)
    # model.test('bla', annotated=True)
    sentence1 = ' '.join([word[0] for word in parsed_sentences[0]])
    results_tag, inference_time = model.infer(sentence1)
    tagged_sentence1 = [f'{sentence1.split(" ")[i]}_{results_tag[i]}' for i in range(len(results_tag))]
    # print(f'results({inference_time}[sec]: {" ".join(tagged_sentence1)}')
    print(f'results: time - {"{0:.2f}".format(inference_time)}[sec], tags - {" ".join(tagged_sentence1)}')