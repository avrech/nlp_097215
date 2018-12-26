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

    def __init__(self, sentences, w2i, i2w, t2i, i2t, l2=0.1, verbose=1):
        '''

        :param sentences: The whole corpus in integer representation.
        :param w2i: word2int function. maps all strings which contains numbers to 1.
        :param i2w: list int2word
        :param t2i: dict of all possible tags.
        :param i2t: list int2tag
        :param wtp: word-tag-pairs dictionaries.
        '''
        # prepare data and get statistics
        print('{} - processing data'.format(datetime.datetime.now()))
        self.verbose = verbose
        self.sentences = sentences # integer representation
        # conversions:
        self.w2i = w2i
        self.i2w = i2w
        self.t2i = t2i
        self.i2t = i2t
        self.wtp = None # word_tag pairs.
        self.tagrams = None
        # features counts & offset:
        self.wtp_offset = None
        self.unigram_offset = None
        self.bigram_offset = None
        self.trigram_offset = None
        self.pref_offset = None
        self.suff_offset = None

        self.feature_set1_offset = None

        self.text_stats = self.get_text_statistics()
        self.tags = list(self.text_stats['tag_count'].keys())
        print('{} - preparing features'.format(datetime.datetime.now()))
        self.num_features = 0
        self.feature_set = []
        self.feature_set1 = []

        self.pref_th = {1: 1000, 2: 1000, 3: 500, 4: 500}
        self.suff_th = {1: 100, 2: 100, 3: 100, 4: 100}
        self.enriched_tags = [None] + self.tags
        self.word_vectors = None
        self.parameter_vector = None
        self.l2 = l2
        self.empirical_counts = None
        self.word_positive_indices = None
        self.use_vector_form = False
        self.l_counter = self.l_grad_counter = 0
        self.safe_softmax = True

        self.get_features()

    @staticmethod
    def safe_add(curr_dict, key):
        curr_dict[key] = curr_dict.get(key, 0) + 1
        # if key not in curr_dict:
        #     curr_dict[key] = 1
        # else:
        #     curr_dict[key] += 1

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

        selected_prefix_tag_pairs = {}
        selected_suffix_tag_pairs = {}
        tagrams = [[unigram_idx, {}] for unigram_idx in range(len(self.i2t))] # adding offset afterward

        for s_idx, sentence in enumerate(self.sentences):
            if np.mod(s_idx,500) == 0:
                print('processing the {}th sentence'.format(s_idx))
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
                if i > 0:
                    self.safe_add(tag_bigrams_counts, (prev_tag, tag))
                self.safe_add(word_trigrams_counts, (prev_prev_word, prev_word, word))
                if i > 1:
                    self.safe_add(tag_trigrams_counts, (prev_prev_tag, prev_tag, tag))


                # convert word to string to process charachter features:
                word = self.i2w[word]
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

        ''' Assign feature index for each word-tag pair'''
        self.wtp_offset = 0
        wtp = [{}] * word_unigrams_counts.keys().__len__()
        for wtp_idx, (w, t) in enumerate(word_tag_counts.keys()):
            wtp[w][t] = wtp_idx + self.wtp_offset
        self.wtp = wtp

        ''' assign feature index for tags n-grams'''
        self.unigram_offset = word_tag_counts.keys().__len__()
        for unigram_idx in range(tagrams.__len__()):
            tagrams[unigram_idx][0] += self.unigram_offset
        self.bigram_offset  = self.unigram_offset + tag_unigrams_counts.keys().__len__()
        self.trigram_offset = self.bigram_offset  + tag_bigrams_counts.keys().__len__()

        # set the bigrams features indices in the feature vector:
        for bigram_idx, (pt, t) in enumerate(tag_bigrams_counts.keys()):
            tagrams[t][1][pt] = [bigram_idx + self.bigram_offset, {}]
        # set the trigrams features indices in the feature vector:
        for trigram_idx, (ppt, pt, t) in enumerate(tag_trigrams_counts.keys()):
            tagrams[t][1][pt][1][ppt] = trigram_idx + self.trigram_offset

        self.tagrams = tagrams
        ''' Filter prefix and suffix'''
        self.pref_th = {1: 1000, 2: 500, 3: 100, 4: 100}
        self.suff_th = {1: 100, 2: 100, 3: 100, 4: 100}

        self.pref_offset = self.trigram_offset + len(tag_trigrams_counts.keys())
        pref_idx = 0
        selected_prefix_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for (p_t, count) in prefix_tag_counter.items():
            if count > self.pref_th[len(p_t[0])]:
                selected_prefix_tag_pairs[p_t] = pref_idx + self.pref_offset
                pref_idx += 1
                selected_prefix_counts[len(p_t[0])] += 1

        self.suff_offset = self.pref_offset + len(selected_prefix_tag_pairs.keys())
        suff_idx = 0
        selected_suffix_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for (s_t, count) in suffix_tag_counter.items():
            if count > self.suff_th[len(s_t[0])]:
                selected_suffix_tag_pairs[s_t] = suff_idx + self.suff_offset
                suff_idx += 1
                selected_suffix_counts[len(p_t[0])] += 1
        if self.verbose == 1:
            print('Total number of prefix features: ', len(selected_prefix_tag_pairs.keys()), 'prefix features counts:', selected_prefix_counts)
            print('Total number of suffix features: ', len(selected_suffix_tag_pairs.keys()), 'suffix features counts:', selected_suffix_counts)
        self.feature_set1_offset = self.suff_offset + len(selected_suffix_tag_pairs.keys())

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
                'next_word_cur_tag': next_word_cur_tag,
                'selected_prefix_tag_pairs': selected_prefix_tag_pairs,
                'selected_suffix_tag_pairs': selected_suffix_tag_pairs
                }

    def get_features(self):
        # define the features set
        # tag-word pairs in dataset (#100) ~500K
        def set_wtp_feature(cntx, pos_features):
            pos = self.wtp[cntx.word].get(cntx.tag)
            if pos is not None:
                pos_features.append(pos)
            return pos_features
        self.feature_set += [set_wtp_feature]

        # self.feature_set += [(lambda w, t: (lambda cntx: 1 if cntx.word == w and cntx.tag == t else 0))(w, t)
        #                      for w, t in self.text_stats['word_tag_pairs'].keys()]

        # suffixes <= 4 and tag pairs in dataset (#101)
        def set_suffix_tag_features(cntx, pos_features):
            word = self.i2w[cntx.word]
            for suff_len in np.arange(1,5):
                s_t = (word[-suff_len:], cntx.tag)
                pos = self.text_stats['selected_suffix_tag_pairs'].get(s_t)
                if pos is not None:
                    pos_features.append(pos)
            return pos_features
        self.feature_set += [set_suffix_tag_features]

        # self.feature_set += [(lambda suff, t: (lambda cntx: 1 if cntx.word.endswith(suff) and
        #                                                          cntx.tag == t else 0))(suff, t)
        #                      for suff, t in self.text_stats['selected_suffix_tag_pairs']]

        # prefixes <= 4 and tag pairs in dataset (#102)
        def set_prefix_tag_features(cntx, pos_features):
            word = self.i2w[cntx.word]
            for pref_len in np.arange(1, 5):
                p_t = (word[:pref_len], cntx.tag)
                pos = self.text_stats['selected_prefix_tag_pairs'].get(p_t)
                if pos is not None:
                    pos_features.append(pos)
            return pos_features

        self.feature_set += [set_prefix_tag_features]

        # self.feature_set += [(lambda pref, t: (lambda cntx: 1 if cntx.word.startswith(pref) and
        #                                                          cntx.tag == t else 0))(pref, t)
        #                      for pref, t in self.text_stats['selected_prefix_tag_pairs']]

        def set_tagrams_features(cntx, pos_features):
            pos_features.append(self.tagrams[cntx.tag][0]) # unigram feature, always exists.
            pos_bigram  = self.tagrams[cntx.tag][1].get(cntx.prev_tag)
            if pos_bigram is None:
                return pos_features
            pos_features.append(pos_bigram[0])
            pos_trigram = self.tagrams[cntx.tag][1].get(cntx.prev_tag)[1].get(cntx.prev_prev_tag)
            if pos_trigram is None:
                return pos_features
            pos_features.append(pos_trigram)
            return pos_features
        self.feature_set += [set_tagrams_features]


        # # tag trigrams in datset (#103) ~8K
        # self.feature_set += [(lambda prev_prev_tag, prev_tag, tag:
        #                       (lambda cntx: 1 if cntx.tag == tag and cntx.prev_tag == prev_tag
        #                                          and cntx.prev_prev_tag == prev_prev_tag else 0))
        #                      (prev_prev_tag, prev_tag, tag)
        #                      for prev_prev_tag, prev_tag, tag in self.text_stats['tag_trigrams'].keys()]
        # # tag bigrams in datset (#104) <1K
        # self.feature_set += [(lambda prev_tag, tag:
        #                       (lambda cntx: 1 if cntx.tag == tag and cntx.prev_tag == prev_tag else 0))(prev_tag, tag)
        #                      for prev_tag, tag in self.text_stats['tag_bigrams'].keys()]
        # # tag unigrams in datset (#105)
        # self.feature_set += [(lambda tag: (lambda cntx: 1 if cntx.tag == tag else 0))(tag) for tag in self.tags]


        # capital first letter tag
        self.feature_set1 += [(lambda t: (lambda cntx: 1 if self.i2w[cntx.word][0].isupper() and cntx.tag == t else 0))(t)
                             for t in self.text_stats['capital_first_letter_tag'].keys()]
        # capital word tag
        self.feature_set1 += [(lambda t: (lambda cntx: 1 if all([letter.isupper() for letter in self.i2w[cntx.word]])
                                                           and cntx.tag == t else 0))(t)
                             for t in self.text_stats['capital_word_tag'].keys()]
        # number tag feature
        self.feature_set1 += [(lambda t: (lambda cntx: 1 if self.i2w[cntx.word].replace('.', '', 1).isdigit() # why 1?
                                                           and cntx.tag == t else 0))(t)
                             for t in self.text_stats['number_tag'].keys()]

        # first word in sentence tag
        self.feature_set1 += [(lambda t: (lambda cntx: 1 if cntx.index == 0 and cntx.tag == t else 0))(t)
                             for t in self.text_stats['first_word_tag'].keys()]

        # second word in sentence tag
        self.feature_set1 += [(lambda t: (lambda cntx: 1 if cntx.index == 1 and cntx.tag == t else 0))(t)
                             for t in self.text_stats['second_word_tag'].keys()]

        # last word in sentence tag
        self.feature_set1 += [(lambda t: (lambda cntx: 1 if cntx.next_word is None and cntx.tag == t else 0))(t)
                             for t in self.text_stats['last_word_tag'].keys()]

        def set_all_other_features(cntx, pos_features):
            pos_features += list(np.array([f(cntx) for f in self.feature_set1]).nonzero()[0] + self.feature_set1_offset)
            return pos_features
        self.feature_set += [set_all_other_features]
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
        self.num_features = self.feature_set1_offset + len(self.feature_set1)
        return self.feature_set, self.tags

    def train_model(self, param_vec=None):
        print('{} - start training'.format(datetime.datetime.now()))
        t_start = time.time()
        # for each word in the corpus, find its feature vector
        word_vectors = []
        word_positive_indices = list() # a list of the features of each senence in the corpus.
        for s_idx, sentence in enumerate(self.sentences):
            positive_indices = list() # a list of the positive features of each word in the current sentence.
            for w_idx in range(len(sentence)):
                context = Context.get_context_tagged(sentence, w_idx)
                positive_indices.append(self.get_positive_features_for_context(context))
            word_positive_indices.append(positive_indices) # a list of the features of each sentence.

        self.word_positive_indices = word_positive_indices # wpi[s_idx][w_idx] is the positive features of a word in a sentence
        self.empirical_counts = self.get_empirical_counts_from_dict()

        # calculate the parameters vector
        print('{} - finding parameter vector'.format(datetime.datetime.now()))
        if param_vec is None:
            param_vec = scipy.optimize.minimize(fun=self.l, x0=np.ones(self.num_features), method='L-BFGS-B',
                                                jac=self.grad_l, options={'maxiter': 17, 'maxfun': 20})

        self.parameter_vector = param_vec.x
        print(self.parameter_vector)
        print('{} - model train complete'.format(datetime.datetime.now()))
        return time.time() - t_start

    # use Viterbi to infer tags for the target sentence
    def infer(self, sentence):
        '''
        Parse the input, convert words to integers (treat all numbers as 1).
        Infer the probable tags.
        TODO: how to treat UNKNOWN WORDS?
        :param sentence: string of words, separated by space chars. example: 'The dog barks'
        :return: a list of corresponding tags - for example: [DT, NN, Vt]
        '''
        print(f'{datetime.datetime.now()} - predict for {sentence}')
        t_start = time.time()
        parsed_sentence_str = [word.rstrip() for word in sentence.split(' ')]
        parsed_sentence = [self.w2i(w) for w in parsed_sentence_str]
        pi = {(0, None, None): 1}
        bp = {}
        for word_index in range(1, len(parsed_sentence) + 1):
            context_dict = {}
            norm_for_context_and_expmax = {}
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
                                        (word_index, prev_tag, prev_prev_tag) not in norm_for_context_and_expmax:
                            if (prev_tag, prev_prev_tag) not in context_dict:
                                context_dict[(prev_tag, prev_prev_tag)] = \
                                    Context.get_context_untagged(parsed_sentence, word_index - 1, tag, prev_tag,
                                                                 prev_prev_tag)
                            norm_for_context_and_expmax[(word_index, prev_tag, prev_prev_tag)] = \
                                self.get_context_norm_and_expmax(context_dict[(prev_tag, prev_prev_tag)])
                    transition_proba = \
                        {prev_prev_tag: self.get_tag_proba(tag, context_dict[(prev_tag, prev_prev_tag)],
                                                           norm=norm_for_context_and_expmax[(word_index, prev_tag, prev_prev_tag)])
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
        result_tags_str = [self.i2t[i] for i in result_tags]
        return result_tags_str, infer_time

    def get_feature_vector_for_context(self, context):
        vector = [feature(context) for feature in self.feature_set]
        return vector

    def get_positive_features_for_context(self, context):
        pos_features = []
        for f in self.feature_set:
            pos_features = f(context,pos_features)
        return np.array(pos_features)

    def get_empirical_counts_from_dict(self):
        emprical_counts = np.zeros(self.num_features)
        for positive_features in self.word_positive_indices:
            for positive_indices in positive_features:
                emprical_counts[positive_indices] += 1
        return emprical_counts

    def get_dot_product(self, feature_vector):
        dot_product = np.sum(np.take(self.parameter_vector,feature_vector))
        return dot_product

    @staticmethod
    def get_dot_product_from_positive_features(positive_indices, v):
        return np.sum(np.take(v, positive_indices))

    # safe-softmax
    def get_tag_proba(self, tag, context, norm=None):
        context.tag = tag
        tag_vector = self.get_positive_features_for_context(context)
        if norm is None:
            norm = self.get_context_norm_and_expmax(context)
        numerator = np.exp(self.get_dot_product(tag_vector)-norm[1])
        proba = numerator / norm[0] if norm[0] > 0 else 0
        return proba

    def get_context_norm_and_expmax(self, context):
        dot_products = []
        for curr_tag in self.tags:
            context.tag = curr_tag
            tag_vector = self.get_positive_features_for_context(context)
            dot_products.append(self.get_dot_product(tag_vector))
        expmax = max(dot_products)
        norm = sum(np.exp(np.array(dot_products)-expmax))
        return norm, expmax

    def log(self, msg):
        if self.verbose:
            print(msg)

    # the ML estimate maximization function
    def l(self, v):
        print('optimizer iter no.', self.l_counter)
        # proba part, the linear term in l(v)
        proba = 0
        # normalization part, i.e. sum_on_x(i){ log( sum_on_y'[exp(v*f(x(i),y'] ) }
        norm_part = 0
        for s_idx, sentence in enumerate(self.sentences):
            for w_idx in range(len(sentence)):
                curr_context = Context.get_context_tagged(sentence, w_idx)
                curr_exp = 0
                # if True:  # self.safe_softmax:
                dot_products = []
                for tag in self.tags:
                    curr_context.tag = tag
                    positive_features = self.get_positive_features_for_context(curr_context)
                    dot_products.append(self.get_dot_product_from_positive_features(positive_features, v))
                safety_term = max(dot_products)
                exponents = np.exp(np.array(dot_products) - safety_term)
                p = self.get_dot_product_from_positive_features(self.word_positive_indices[s_idx][w_idx], v) - safety_term
                proba += p
                norm_part += np.log(sum(exponents))
        res = proba - norm_part - 0.5 * self.l2 * v.T @ v # regularization l2

        self.log(f'l = {self.l_counter},{res}')
        self.l_counter += 1
        # return minus res because the optimizer does not know to maximize,
        # so we minimize (-l(v))
        return -res

    def grad_l(self, v):
        # expected counts
        expected_counts = np.zeros(v.shape)
        for s_idx, sentence in enumerate(self.sentences):
            for w_idx in range(len(sentence)):
                curr_context = Context.get_context_tagged(sentence, w_idx)
                f_xi_for_all_ytag = []       # the features f(x(i),y') for y' in Y
                dot_products = []  # f(x(i),y') @ v for y' in Y
                # safe softmax: first calculate all dot products, then deduce the max val to avoid overflow
                for tag in self.tags:
                    curr_context.tag = tag
                    f_xi_ytag = self.get_positive_features_for_context(curr_context)
                    f_xi_for_all_ytag.append(f_xi_ytag)
                    dot_products.append(self.get_dot_product_from_positive_features(f_xi_ytag, v))
                dot_products = np.array(dot_products) - max(dot_products) # make the exponent safe.

                exponents = np.exp2(dot_products)
                softmax_denominator = sum(exponents)
                softmax_xi_ytag = exponents / softmax_denominator

                # the nominator cannot be explicitely computed in positive feature representation.
                # the update is element-wise:
                if softmax_denominator > 0:
                    for tag, pos_f in enumerate(f_xi_for_all_ytag):
                        for v_idx in pos_f:
                            expected_counts[v_idx] -= softmax_xi_ytag[tag]


        res = self.empirical_counts - expected_counts - self.l2 * v # regularization l2
        self.log(f'grad = {self.l_grad_counter}')
        self.l_grad_counter += 1
        return -res # we return minus res because the optimizer knows only to minimize (-l(v))


    def test(self, text, annotated=False):
        pass

def is_number(word):
    '''
    consider any string which contains digits as a number.
    TODO: maybe only the chars: 0 1 2 3 4 5 6 7 8 9 0 , . - + e ?
    :param word: string - example "400,000", "3.1", "1e-10"
    :return: true/ false
    '''
    for char in word:
        if char.isdigit():
            return 1
    return 0

def num2one(word):
    return "1" if is_number(word) else word

# receives input file (of tagged sentences),
# returns a list of sentences, each sentence is a list of (word,tag)s
# also performs base verification of input
def get_parsed_sentences_from_tagged_file(filename, n=None):
    '''
    convert words and tags to int.
    :param      filename:
    :return:    sentences - parsed_sentences in integer format.
                word2int - dict
                int2word - list
                tag2int - dict
                int2tag - list
    '''
    sentences_int = []
    word_tag_pairs = None
    tag2int = dict()
    int2tag = list()
    word2int = dict()
    int2word = list()
    f_text = None
    print(f'{datetime.datetime.now()} - loading data from file {filename}')
    with open(filename) as f:
    # f_text = open(filename)
        if n is not None:
            f_text = [next(f) for _ in range(n)]
        else:
            f_text = [row for row in f]

    all_words_and_tags = []
    sentences = []
    for row in f_text:
        tagged_words = [word.rstrip() for word in row.split(' ')]
        # TODO: here I replace all words which contain digits to '1'. is it good?
        words_and_tags = tuple([tuple([num2one(tagged_word.split('_')[0]),tagged_word.split('_')[1]]) for tagged_word in tagged_words])
        sentences.append(words_and_tags)
        all_words_and_tags += words_and_tags
    bad_words = [word for word in all_words_and_tags if len(word) != 2]
    if bad_words:
        print(f'found {len(bad_words)} bad words - {bad_words}')

    # Generate dictionaries for word/tag - int conversion:
    all_words_set = set([word[0] for word in all_words_and_tags])
    for idx, w in enumerate(all_words_set):
        word2int[w] = idx
        int2word.append(w)

    all_tags_set = set([word[1] for word in all_words_and_tags])
    for idx, t in enumerate(all_tags_set):
        tag2int[t] = idx
        int2tag.append(t)


    all_words_and_tags_int = []
    w2i = lambda w: word2int.get(num2one(w), -1)
    for row in f_text:
        tagged_words = [word.rstrip() for word in row.split(' ')]
        words_and_tags_int = tuple([(w2i(tagged_word.split('_')[0]), tag2int[tagged_word.split('_')[1]])
                                    for tagged_word in tagged_words])
        sentences_int.append(words_and_tags_int)
        all_words_and_tags_int += words_and_tags_int

    print(f'{datetime.datetime.now()} - found {len(all_tags_set)} tags - {all_tags_set}')

    return sentences_int, w2i, int2word, tag2int, int2tag

def parse_test_set(filename, n=None):
    '''
    parse input file to sentences. convert words and tags to int according to the training corpus.
    :param      filename:
    :return:    sentences - parsed_sentences in integer format.
    '''
    sentences = []
    f_text = None
    print(f'{datetime.datetime.now()} - loading data from file {filename}')
    with open(filename) as f:
        if n is not None:
            f_text = [next(f) for _ in range(n)]
        else:
            f_text = [row for row in f]

    for row in f_text:
        tagged_words = [word.rstrip() for word in row.split(' ')]
        words_and_tags = tuple([(tagged_word.split('_')) for tagged_word in tagged_words])
        sentences.append(words_and_tags)
    return sentences

def evaluate(model, testset_file, n_samples=1, max_words=None):
    parsed_testset = parse_test_set(testset_file, n=n_samples) # TODO: use trainset dictionary for test set.
    predictions = []
    accuracy = []

    for s_idx, sample in enumerate(parsed_testset):
        sentence = ' '.join([word_id[0] for word_id in sample[:max_words]])
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

    my_model = MEMM(parsed_sentences, w2i, i2w, t2i, i2t)
    train_time = my_model.train_model()
    print(f'train: time - {"{0:.2f}".format(train_time)}[sec]')
    with open('model_prm.pkl', 'wb') as f:
        pickle.dump(my_model.parameter_vector, f)
    # f = open('model_prm.pkl', 'rb')
    # param_vector = pickle.load(f)
    # model.test('bla', annotated=True)



    evaluate(my_model, 'train.wtag', max_words=10)

    # Evaluate test set:
    evaluate(my_model, 'test.wtag', 1, 5)
