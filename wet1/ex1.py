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
        self.index = len(history)
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
    BEAM_MIN = 10

    def __init__(self, sentences, w2i, i2w, t2i, i2t):
        # prepare data and get statistics
        print('{} - processing data'.format(datetime.datetime.now()))
        self.sentences = sentences
        self.w2i = w2i
        self.i2w = i2w
        self.t2i = t2i
        self.i2t - i2t
        self.text_stats = self.get_text_statistics()
        self.tags = list(self.text_stats['tag_count'].keys())
        print('{} - preparing features'.format(datetime.datetime.now()))
        self.num_features = 0
        self.feature_set = []
        self.pref_th = {1: 1000, 2: 1000, 3: 500, 4: 500}
        self.suff_th = {1: 100, 2: 100, 3: 100, 4: 100}
        self.get_features()
        self.enriched_tags = [None] + self.tags
        self.feature_set1 = []
        self.word_vectors = None
        self.parameter_vector = None
        self.empirical_counts = None
        self.word_positive_indices = None
        self.use_vector_form = False
        self.l_counter = self.l_grad_counter = 0
        self.safe_softmax = True
        self.verbose = 1

    @staticmethod
    def safe_add(curr_dict, key):
        if key not in curr_dict:
            curr_dict[key] = 1
        else:
            curr_dict[key] += 1

    def get_text_statistics(self):
        print('{} - getting text stats'.format(datetime.datetime.now()))
        word_tag_counts = {}
        word_unigrams_counts = {}
        tag_unigrams_counts = {}
        tag_bigrams_counts = {}
        word_bigrams_counts = {}
        tag_trigrams_counts = {}
        word_trigrams_counts = {}
        prefix_tag_counter = {}
        suffix_tag_counter = {}
        capital_first_letter_tag = {}
        capital_word_tag = {}
        number_tag = {}
        first_word_tag = {}
        second_word_tag = {}
        last_word_tag = {}
        prev_word_cur_tag = {}
        next_word_cur_tag = {}

        for sentence in self.sentences:
            for i, word_tag in enumerate(sentence):

                word = word_tag[0]
                prev_word = sentence[i - 1][0] if i > 0 else None
                prev_prev_word = sentence[i - 2][0] if i > 1 else None

                tag = word_tag[1]
                prev_tag = sentence[i - 1][1] if i > 0 else None
                prev_prev_tag = sentence[i - 2][1] if i > 1 else None

                self.safe_add(word_unigrams_counts, word)
                self.safe_add(tag_unigrams_counts, tag)
                self.safe_add(word_tag_counts, word_tag)
                self.safe_add(word_bigrams_counts, (prev_word, word))
                self.safe_add(tag_bigrams_counts, (prev_tag, tag))
                self.safe_add(word_trigrams_counts, (prev_prev_word, prev_word, word))
                self.safe_add(tag_trigrams_counts, (prev_prev_tag, prev_tag, tag))

                self.safe_add(prefix_tag_counter, (word[:1], tag))
                if len(word) > 1:
                    self.safe_add(suffix_tag_counter, (word[-1:], tag))
                    self.safe_add(prefix_tag_counter, (word[:2], tag))
                if len(word) > 2:
                    self.safe_add(suffix_tag_counter, (word[-2:], tag))
                    self.safe_add(prefix_tag_counter, (word[:3], tag))
                if len(word) > 3:
                    self.safe_add(suffix_tag_counter, (word[-3:], tag))
                    self.safe_add(prefix_tag_counter, (word[:4], tag))
                if len(word) > 4:
                    self.safe_add(suffix_tag_counter, (word[-4:], tag))

                if word[0].isupper():
                    self.safe_add(capital_first_letter_tag, tag)
                if all([letter.isupper() for letter in word]):
                    self.safe_add(capital_word_tag, tag)
                if word.replace('.', '', 1).isdigit():
                    self.safe_add(number_tag, tag)

                if i == 1:
                    self.safe_add(first_word_tag, tag)
                if i == 2:
                    self.safe_add(second_word_tag, tag)
                if i == len(sentence) - 1:
                    self.safe_add(last_word_tag, tag)
                if i > 0:
                    self.safe_add(prev_word_cur_tag, (prev_word, tag))
                if i < len(sentence)-1:
                    self.safe_add(next_word_cur_tag, (sentence[i + 1][0], tag))

        return {'word_count': word_unigrams_counts,
                'tag_count': tag_unigrams_counts,
                'word_tag_pairs': word_tag_counts,
                'word_bigrams': word_bigrams_counts,
                'tag_bigrams': tag_bigrams_counts,
                'word_trigrams': word_trigrams_counts,
                'tag_trigrams': tag_trigrams_counts,
                'prefix_tag': prefix_tag_counter,
                'suffix_tag': suffix_tag_counter,
                'capital_first_letter_tag': capital_first_letter_tag,
                'capital_word_tag': capital_word_tag,
                'number_tag': number_tag,
                'first_word_tag': first_word_tag,
                'second_word_tag': second_word_tag,
                'last_word_tag': last_word_tag,
                'prev_word_cur_tag': prev_word_cur_tag,
                'next_word_cur_tag': next_word_cur_tag
                }

    def get_features(self):
        # define the features set
        # tag-word pairs in dataset (#100) ~15K
        self.feature_set += [(lambda w, t: (lambda cntx: 1 if cntx.word == w and cntx.tag == t else 0))(w, t)
                             for w, t in self.text_stats['word_tag_pairs'].keys()]
        ''' Filter prefix and suffix'''
        # # Define thresholds:
        # pref_th = {1: 1000, 2: 1000, 3: 500, 4: 500}
        # selected_prefix = {1: [], 2: [], 3: [], 4: []}
        # # print('prefix counts:')
        # for l in np.arange(1, 5):
        #     for p in prefix[l].keys():
        #         if prefix[l][p] > pref_th[l]:
        #             selected_prefix[l].append(p)
        #             # print(p, ': ', prefix[l][p])
        #
        # suff_th = {1: 100, 2: 100, 3: 100, 4: 100}
        # selected_suffix = {1: [], 2: [], 3: [], 4: []}
        # # print('suffix counts:')
        # for l in np.arange(1, 5):
        #     for p in suffix[l].keys():
        #         if suffix[l][p] > suff_th[l]:
        #             selected_suffix[l].append(p)
        #             # print(p, ': ', suffix[l][p])
        #
        # print('# prefix features: ',
        #       '-1', selected_prefix[1].__len__(),
        #       '-2', selected_prefix[2].__len__(),
        #       '-3', selected_prefix[3].__len__(),
        #       '-4', selected_prefix[4].__len__(),
        #       ' | Total-', sum([p.__len__() for p in selected_prefix.values()]))
        # print('# suffix features: ',
        #       '-1', selected_suffix[1].__len__(),
        #       '-2', selected_suffix[2].__len__(),
        #       '-3', selected_suffix[3].__len__(),
        #       '-4', selected_suffix[4].__len__(),
        #       ' | Total-', sum([p.__len__() for p in selected_suffix.values()]))
        #
        self.pref_th = {1: 1000, 2: 500, 3: 100, 4: 100}
        self.suff_th = {1: 100, 2: 100, 3: 100, 4: 100}

        selected_suffix_tag_pairs = [s_t for (s_t, count) in self.text_stats['suffix_tag'].items() if
                                     count > self.suff_th[len(s_t[0])]]
        selected_prefix_tag_pairs = [p_t for (p_t, count) in self.text_stats['prefix_tag'].items() if
                                     count > self.pref_th[len(p_t[0])]]

        # suffixes <= 4 and tag pairs in dataset (#101)
        self.feature_set += [(lambda suff, t: (lambda cntx: 1 if cntx.word.endswith(suff) and
                                                                 cntx.tag == t else 0))(suff, t)
                             for suff, t in selected_suffix_tag_pairs]
        # prefixes <= 4 and tag pairs in dataset (#102)
        self.feature_set += [(lambda pref, t: (lambda cntx: 1 if cntx.word.startswith(pref) and
                                                                 cntx.tag == t else 0))(pref, t)
                             for pref, t in selected_prefix_tag_pairs]
        # tag trigrams in datset (#103) ~8K
        self.feature_set += [(lambda prev_prev_tag, prev_tag, tag:
                              (lambda cntx: 1 if cntx.tag == tag and cntx.prev_tag == prev_tag
                                                 and cntx.prev_prev_tag == prev_prev_tag else 0))
                             (prev_prev_tag, prev_tag, tag)
                             for prev_prev_tag, prev_tag, tag in self.text_stats['tag_trigrams'].keys()]
        # tag bigrams in datset (#104) <1K
        self.feature_set += [(lambda prev_tag, tag:
                              (lambda cntx: 1 if cntx.tag == tag and cntx.prev_tag == prev_tag else 0))(prev_tag, tag)
                             for prev_tag, tag in self.text_stats['tag_bigrams'].keys()]
        # tag unigrams in datset (#105)
        self.feature_set += [(lambda tag: (lambda cntx: 1 if cntx.tag == tag else 0))(tag) for tag in self.tags]

        # capital first letter tag
        self.feature_set += [(lambda t: (lambda cntx: 1 if cntx.word[0].isupper() and cntx.tag == t else 0))(t)
                             for t in self.text_stats['capital_first_letter_tag'].keys()]
        # capital word tag
        self.feature_set += [(lambda t: (lambda cntx: 1 if all([letter.isupper() for letter in cntx.word])
                                                           and cntx.tag == t else 0))(t)
                             for t in self.text_stats['capital_word_tag'].keys()]
        # number tag feature
        self.feature_set += [(lambda t: (lambda cntx: 1 if cntx.word.replace('.', '', 1).isdigit()
                                                           and cntx.tag == t else 0))(t)
                             for t in self.text_stats['number_tag'].keys()]

        # first word in sentence tag
        self.feature_set += [(lambda t: (lambda cntx: 1 if cntx.index == 0 and cntx.tag == t else 0))(t)
                             for t in self.text_stats['first_word_tag'].keys()]

        # second word in sentence tag
        self.feature_set += [(lambda t: (lambda cntx: 1 if cntx.index == 0 and cntx.tag == t else 0))(t)
                             for t in self.text_stats['second_word_tag'].keys()]

        # last word in sentence tag
        self.feature_set += [(lambda t: (lambda cntx: 1 if cntx.index == 0 and cntx.tag == t else 0))(t)
                             for t in self.text_stats['last_word_tag'].keys()]

        # # previous word + current tag pairs (#106) ~23K
        # self.feature_set += [(lambda pw, t: (lambda cntx: 1 if cntx.history[-1] is not None and
        #                                                        cntx.history[-1][0] == pw and
        #                                                        cntx.tag == t else 0))(pw, t)
        #                      for (pw, t) in self.text_stats['prev_word_cur_tag'].keys()]
        #
        # # next word + current tag pairs (#107) ~ 30K
        # self.feature_set += [(lambda nw, t: (lambda cntx: 1 if cntx.history[-1] is not None and
        #                                                        cntx.history[-1][0] == nw and
        #                                                        cntx.tag == t else 0))(nw, t)
        #                      for (nw, t) in self.text_stats['prev_word_cur_tag'].keys()]

        return self.feature_set, self.tags

    def train_model(self, param_vec=None):
        print('{} - start training'.format(datetime.datetime.now()))
        t_start = time.time()
        # for each word in the corpus, find its feature vector
        word_vectors = []
        word_positive_indices = list()
        for s_idx, sentence in enumerate(self.sentences):
            positive_indices = list() # a list of the features of each word in the current sentence.
            for w_idx in range(len(sentence)):
                context = Context.get_context_tagged(sentence, w_idx)
                if self.use_vector_form:
                    vector = self.get_feature_vector_for_context(context)
                    word_vectors.append(vector)
                else:
                    positive_indices.append(self.get_positive_features_for_context(context))
            if not self.use_vector_form:
                word_positive_indices.append(positive_indices) # a list of the features of each sentence.

        if self.use_vector_form:
            self.word_vectors = np.array(word_vectors)
            self.empirical_counts = np.sum(self.word_vectors, axis=0)
        else:
            self.word_positive_indices = word_positive_indices
            self.empirical_counts = self.get_empirical_counts_from_dict()

        # calculate the parameters vector
        print('{} - finding parameter vector'.format(datetime.datetime.now()))
        if param_vec is None:
            if self.use_vector_form:
                param_vec = scipy.optimize.minimize(fun=self.l_vector, x0=np.ones(len(self.feature_set)),
                                                    method='L-BFGS-B', jac=self.grad_l_vector,
                                                    options={'maxiter': 17, 'maxfun': 20})
            else:
                param_vec = scipy.optimize.minimize(fun=self.l, x0=np.ones(len(self.feature_set)), method='L-BFGS-B',
                                                    jac=self.grad_l, options={'maxiter': 17, 'maxfun': 20})
        self.parameter_vector = param_vec.x
        print(self.parameter_vector)
        print('{} - model train complete'.format(datetime.datetime.now()))
        return time.time() - t_start

    # use Viterbi to infer tags for the target sentence
    def infer(self, sentence):
        print(f'{datetime.datetime.now()} - predict for {sentence}')
        t_start = time.time()
        parsed_sentence = [word.rstrip() for word in sentence.split(' ')]
        pi = {(0, None, None): 1}
        bp = {}
        for word_index in range(1, len(parsed_sentence) + 1):
            context_dict = {}
            norm_for_context = {}
            pi_temp = {}
            bp_temp = {}
            for tag in self.tags:
                for prev_tag in self.enriched_tags:

                    past_proba = {prev_prev_tag: pi.get((word_index - 1, prev_prev_tag, prev_tag), 0) for prev_prev_tag
                                  in self.enriched_tags}
                    # save the context and the norm for the context, prev tag, prev prev tag once
                    # (otherwise we calculate it many times for different tags
                    for prev_prev_tag in self.enriched_tags:
                        if past_proba[prev_prev_tag] != 0 and\
                                        (word_index, prev_tag, prev_prev_tag) not in norm_for_context:
                            if (prev_tag, prev_prev_tag) not in context_dict:
                                context_dict[(prev_tag, prev_prev_tag)] = \
                                    Context.get_context_untagged(parsed_sentence, word_index - 1, tag, prev_tag,
                                                                 prev_prev_tag)
                            norm_for_context[(word_index, prev_tag, prev_prev_tag)] = \
                                self.get_context_norm(context_dict[(prev_tag, prev_prev_tag)])
                    transition_proba = \
                        {prev_prev_tag: self.get_tag_proba(tag, context_dict[(prev_tag, prev_prev_tag)],
                                                           norm=norm_for_context[(word_index, prev_tag, prev_prev_tag)])
                         for prev_prev_tag in self.enriched_tags if past_proba[prev_prev_tag] != 0}

                    pi_candidates = [past_proba.get(prev_prev_tag, 0) * transition_proba.get(prev_prev_tag, 0)
                                     for prev_prev_tag in self.enriched_tags]
                    pi_temp[(word_index, prev_tag, tag)] = max(pi_candidates)
                    bp_temp[(word_index, prev_tag, tag)] = np.argmax(pi_candidates)

            # trim entries with low probability (beam search)
            # find the probabilty above which we have at least BEAM_MIN possibilities
            min_proba_for_stage = 1
            count_entries_per_proba = len([pi_val for pi_key, pi_val in pi_temp.items()
                                           if pi_val >= min_proba_for_stage])

            while count_entries_per_proba < self.BEAM_MIN:
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

    def get_positive_features_for_context(self, context):
        return np.array([feature(context) for feature in self.feature_set]).nonzero()

    def get_empirical_counts_from_dict(self):
        emprical_counts = np.zeros(len(self.feature_set))
        for positive_features in self.word_positive_indices:
            for positive_indices in positive_features:
                emprical_counts[positive_indices] += 1
        return emprical_counts

    def get_dot_product(self, feature_vector):
        dot_product = sum([feature_vector[i] * self.parameter_vector[i] for i in range(len(feature_vector))])
        return dot_product

    @staticmethod
    def get_dot_product_from_positive_features(positive_indices, v):
        return np.sum(np.take(v, positive_indices))

    # softmax
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

    def log(self, msg):
        if self.verbose:
            print(msg)

    # the ML estimate maximization function
    def l(self, v):
        print('optimizer iter no.', self.l_counter)
        # proba part
        proba = 0
        # normalization part
        norm_part = 0
        for s_idx, sentence in enumerate(self.sentences):
            for w_idx in range(len(sentence)):
                proba += self.get_dot_product_from_positive_features(self.word_positive_indices[s_idx][w_idx], v)
                curr_context = Context.get_context_tagged(sentence, w_idx)
                curr_exp = 0
                if False:  # self.safe_softmax:
                    dot_products = []
                    for tag in self.tags:
                        curr_context.tag = tag
                        # vector = self.get_feature_vector_for_context(curr_context)
                        positive_features = self.get_positive_features_for_context(curr_context)
                        dot_products.append(self.get_dot_product_from_positive_features(positive_features, v))
                    dot_products = np.array(dot_products) - max(dot_products)
                    for val in dot_products:
                        curr_exp += 2 ** val
                else:
                    for tag in self.tags:
                        curr_context.tag = tag
                        # vector = self.get_feature_vector_for_context(curr_context)
                        positive_features = self.get_positive_features_for_context(curr_context)
                        curr_exp += 2 ** self.get_dot_product_from_positive_features(positive_features, v)
                norm_part += math.log(curr_exp, 2)
        res = proba - norm_part
        self.log(f'l = {self.l_counter},{res}')
        self.l_counter += 1
        return -res

    # the ML estimate maximization function
    def l_vector(self, v):
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
        res = proba - norm_part
        self.log(f'l = {self.l_counter},{res}')
        self.l_counter += 1
        return -res

    def grad_l(self, v):
        # expected counts
        expected_counts = 0
        for s_idx, sentence in enumerate(self.sentences):
            for w_idx in range(len(sentence)):
                curr_context = Context.get_context_tagged(sentence, w_idx)
                normalization = 0
                nominator = 0
                if self.safe_softmax:
                    dot_products = []
                    vectors = []
                    # safe softmax: first calculate all dot products, then deduce the max val to avoid overflow
                    for tag in self.tags:
                        curr_context.tag = tag
                        vector = self.get_feature_vector_for_context(curr_context)
                        vectors.append(vector)
                        curr_positive_features = self.word_positive_indices[s_idx][w_idx]
                        dot_products.append(self.get_dot_product_from_positive_features(curr_positive_features, v))
                    dot_products = np.array(dot_products) - max(dot_products)
                    for j, product in enumerate(dot_products):
                        curr_exp = 2 ** product
                        normalization += curr_exp
                        nominator += np.multiply(vectors[j], curr_exp)
                else:
                    for tag in self.tags:
                        curr_context.tag = tag
                        vector = self.get_feature_vector_for_context(curr_context)
                        curr_positive_features = self.word_positive_indices[s_idx][w_idx]
                        curr_exp = 2 ** self.get_dot_product_from_positive_features(curr_positive_features, v)
                        normalization += curr_exp
                        nominator += np.multiply(vector, curr_exp)

                expected_counts += nominator / normalization if normalization > 0 else 0
        res = self.empirical_counts - expected_counts
        self.log(f'grad = {self.l_grad_counter}')
        self.l_grad_counter += 1
        return -res

    def grad_l_vector(self, v):
        self.log(f'grad = {self.l_grad_counter}')
        self.l_grad_counter += 1
        # expected counts
        expected_counts = 0
        for sentence in self.sentences:
            for i in range(len(sentence)):
                curr_context = Context.get_context_tagged(sentence, i)
                normalization = 0
                nominator = 0
                for tag in self.tags:
                    curr_context.tag = tag
                    vector = self.get_feature_vector_for_context(curr_context)
                    curr_exp = 2 ** np.dot(v, vector)
                    normalization += curr_exp
                    nominator += np.multiply(vector, curr_exp)

                expected_counts += nominator / normalization if normalization > 0 else 0
        res = self.empirical_counts - expected_counts
        self.log(f'grad = {self.l_grad_counter}')
        self.l_grad_counter += 1
        return -res

    def test(self, text, annotated=False):
        pass


# receives input file (of tagged sentences),
# returns a list of sentences, each sentence is a list of (word,tag)s
# also performs base verification of input
def get_parsed_sentences_from_tagged_file(filename):
    '''
    convert words and tags to int.
    :param      filename:
    :return:    sentences - parsed_sentences in integer format.
                word2int - dict
                int2word - list
                tag2int - dict
                int2tag - list
    '''
    print(f'{datetime.datetime.now()} - loading data from file {filename}')
    f_text = open(filename)
    all_words_and_tags = []
    sentences = []
    for row in f_text:
        tagged_words = [word.rstrip() for word in row.split(' ')]
        words_and_tags = tuple([tuple(tagged_word.split('_')) for tagged_word in tagged_words])
        sentences.append(words_and_tags)
        all_words_and_tags += words_and_tags
    bad_words = [word for word in all_words_and_tags if len(word) != 2]
    if bad_words:
        print(f'found {len(bad_words)} bad words - {bad_words}')

    # Generate dictionaries for word/tag - int conversion:
    all_words_set = set([word[0] for word in all_words_and_tags])
    word2int = dict()
    int2word = list()
    for idx, w in enumerate(all_words_set):
        word2int[w] = idx
        int2word.append(w)

    all_tags_set = set([word[1] for word in all_words_and_tags])
    tag2int = dict()
    int2tag = list()
    for idx, t in enumerate(all_tags_set):
        tag2int[t] = idx
        int2tag.append(t)

    # Generate word_tag_options dicts:
    word_tag_pairs = [[]] * len(int2word)
    [word_tag_pairs[word2int[w_t[0]]].append(tag2int[w_t[1]]) for w_t in all_words_and_tags]

    # Convert sentences to integer representation:
    for s_idx, s in enumerate(sentences):
        for w_idx, w_t in enumerate(s):
            sentences[s_idx][w_idx][0] = word2int[w_t[0]]
            sentences[s_idx][w_idx][1] = tag2int[w_t[1]]


    # tags_counter = Counter(all_tags_list)
    # all_words_list = [word[0] for word in all_words_and_tags]
    # words_counter = Counter(all_words_list)
    # all_words_set = set(all_words_list)
    print(f'{datetime.datetime.now()} - found {len(all_tags_set)} tags - {all_tags_set}')
    return sentences, word2int, int2word, tag2int, int2tag


def evaluate(model, testset_file, n_samples, max_words=None):
    parsed_testset = get_parsed_sentences_from_tagged_file(testset_file) # TODO: use trainset dictionary for test set.
    predictions = []
    accuracy = []
    for ii, sample in enumerate(parsed_testset[:n_samples]):
        sentence = ' '.join([word[0] for word in sample[:max_words]])
        results_tag, inference_time = model.infer(sentence)
        predictions.append(results_tag)
        comparison = [results_tag[word_idx] == sample[word_idx][1] for word_idx in range(max_words)]
        accuracy.append(sum(comparison) / len(comparison))
        tagged_sentence = ['{}_{}'.format(sentence.split(" ")[i], results_tag[i]) for i in range(len(results_tag))]
        print(f'results: time - {"{0:.2f}".format(inference_time)}[sec], tags - {" ".join(tagged_sentence)}')
    print(f'average accuracy: {"{0:.2f}".format(np.mean(accuracy))}')

if __name__ == "__main__":
    # load training set
    parsed_sentences, w2i, i2w, t2i, i2t = get_parsed_sentences_from_tagged_file('train.wtag')

    my_model = MEMM(parsed_sentences[:50], w2i, i2w, t2i, i2t)
    train_time = my_model.train_model()
    print(f'train: time - {"{0:.2f}".format(train_time)}[sec]')
    with open('model_prm.pkl', 'wb') as f:
        pickle.dump(my_model.parameter_vector, f)
    # f = open('model_prm.pkl', 'rb')
    # param_vector = pickle.load(f)
    # model.test('bla', annotated=True)

    evaluate(my_model, 'train.wtag', 1, 5)

    # Evaluate test set:
    evaluate(my_model, 'test.wtag', 1, 5)
