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

    def __init__(self, sentences, w2i, i2w, t2i, i2t, l2=0.1, verbose=1, beam_search=2):
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
        self.cflt_offset = None
        self.cwt_offset = None
        self.nt_offset  = None
        self.fwt_offset = None
        self.swt_offset = None
        self.lwt_offset = None

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
        self.training_features = [None] * len(self.sentences)
        self.softmax_dict = {} # for inference
        self.beam_search = beam_search
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

        ''' assign feature index for capital first letter tag'''
        cflt = {}
        self.cflt_offset = self.suff_offset + len(selected_suffix_tag_pairs.keys())
        for idx, key in enumerate(capital_first_letter_tag.keys()):
            cflt[key] = idx + self.cflt_offset

        ''' assign feature index for capital word tag'''
        cwt = {}
        self.cwt_offset = self.cflt_offset + len(cflt.keys())
        for idx, key in enumerate(capital_word_tag.keys()):
            cwt[key] = idx + self.cwt_offset

        ''' assign feature index for number tag'''
        nt = {}
        self.nt_offset = self.cwt_offset + len(cwt.keys())
        for idx, key in enumerate(number_tag.keys()):
            nt[key] = idx + self.nt_offset

        ''' assign feature index for first_word_tag'''
        fwt = {}
        self.fwt_offset = self.nt_offset + len(nt.keys())
        for idx, key in enumerate(first_word_tag.keys()):
            fwt[key] = idx + self.fwt_offset

        ''' assign feature index for second_word_tag'''
        swt = {}
        self.swt_offset = self.fwt_offset + len(fwt.keys())
        for idx, key in enumerate(second_word_tag.keys()):
            swt[key] = idx + self.swt_offset

        ''' assign feature index for last_word_tag'''
        lwt = {}
        self.lwt_offset = self.swt_offset + len(swt.keys())
        for idx, key in enumerate(last_word_tag.keys()):
            lwt[key] = idx + self.lwt_offset

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
                'selected_suffix_tag_pairs': selected_suffix_tag_pairs,
                'cflt': cflt,
                'cwt': cwt,
                'nt': nt,
                'fwt': fwt,
                'swt': swt,
                'lwt': lwt
                }

    def get_features(self):
        # define the features set
        # tag-word pairs in dataset (#100) ~15K
        def set_wtp_feature(cntx, pos_features):
            pos = self.wtp[cntx.word].get(cntx.tag)
            if pos is not None:
                pos_features.append(pos)
            return pos_features
        self.feature_set += [set_wtp_feature]

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

        # set first capital letter feature:
        def set_cflt_features(cntx, pos_features):
            if self.i2w[cntx.word][0].isupper():
                pos = self.text_stats['cflt'].get(cntx.tag)
                if pos is not None:
                    pos_features.append(pos)
            return pos_features
        self.feature_set += [set_cflt_features]

        # set capital word feature
        def set_cwt_features(cntx, pos_features):
            if all([letter.isupper() for letter in self.i2w[cntx.word]]):
                pos = self.text_stats['cwt'].get(cntx.tag)
                if pos is not None:
                    pos_features.append(pos)
            return pos_features
        self.feature_set += [set_cwt_features]

        # number tag feature
        def set_nt_features(cntx, pos_features):
            if self.i2w[cntx.word].replace('.', '', 1).isdigit(): # TODO use is_number instead?
                pos = self.text_stats['nt'].get(cntx.tag)
                if pos is not None:
                    pos_features.append(pos)
            return pos_features
        self.feature_set += [set_nt_features]

        # first word in sentence tag
        def set_fwt_features(cntx, pos_features):
            if cntx.index == 0:
                pos = self.text_stats['fwt'].get(cntx.tag)
                if pos is not None:
                    pos_features.append(pos)
            return pos_features
        self.feature_set += [set_fwt_features]

        # second word in sentence tag
        def set_swt_features(cntx, pos_features):
            if cntx.index == 1:
                pos = self.text_stats['swt'].get(cntx.tag)
                if pos is not None:
                    pos_features.append(pos)
            return pos_features
        self.feature_set += [set_swt_features]

        # last word in sentence tag
        def set_lwt_features(cntx, pos_features):
            if cntx.next_word is None:
                pos = self.text_stats['lwt'].get(cntx.tag)
                if pos is not None:
                    pos_features.append(pos)
            return pos_features
        self.feature_set += [set_lwt_features]

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
        self.num_features = self.lwt_offset + len(self.text_stats['lwt'].keys())
        return self.feature_set, self.tags

    def calc_pos_features_for_training(self):
        for s_idx, sentence in enumerate(self.sentences):
            sentence_f = [None] * len(sentence)
            for w_idx in range(len(sentence)):
                word_f = [None] * self.tags.__len__()
                curr_context = Context.get_context_tagged(sentence, w_idx)
                for tag in self.tags:
                    curr_context.tag = tag
                    word_f[tag] = self.get_positive_features_for_context(curr_context)
                sentence_f[w_idx] = word_f
            self.training_features[s_idx] = sentence_f

    def train_model(self, param_vec=None):
        print('Pre-processing - calculate features for training corpus...')
        self.calc_pos_features_for_training()
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
                                                jac=self.grad_l,
                                                options={'maxiter': 50,
                                                         # 'maxls' : 10, # default 20
                                                         # 'ftol'  : 0.05,
                                                         'maxfun': 100,
                                                         # 'maxcor': 10,
                                                         'disp': True})

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
        print(datetime.datetime.now(), ' - predict for ', sentence)
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

    def infer_viterbi_beam_search(self, sentence_str):
        t_start = time.time()
        parsed_sentence_str = [word.rstrip() for word in sentence_str.split(' ')]
        sentence = [self.w2i(w) for w in parsed_sentence_str]
        pi = {(-1, None, None): 1}
        bp = {}
        N = len(sentence)
        S = {idx: self.tags for idx in np.arange(N)}
        S[-2] = [None]
        S[-1] = [(None, None)]  # limited group of candidates for beam-search
        for k in range(N):
            # temp_pi = {}
            # temp_bp = {}
            cur_cntx = Context.get_context_untagged(sentence, k, None, None, None)
            self.softmax_dict = {}
            wuv_probs = {}
            for (w, u) in S[k-1]:
                cur_cntx.prev_tag = u
                cur_cntx.prev_prev_tag = w
                norm = self.get_norm_expmax((w, u, 0), cur_cntx)

                for v in S[k]:
                    # calculate arg/max upon w of
                    # pi[(k-1,w,u)] *
                    # prob of word k to get tag v given context of word k in sentence
                    cur_cntx.tag = v
                    wuv_prob = self.get_tag_proba(v, cur_cntx, norm) * pi[(k - 1, w, u)]  # max term
                    # best_prob = 0
                    # best_w = None
                    # for w in [w for (w, u) in S[k-1]]:
                    #     cur_cntx.prev_prev_tag = w
                    #     w_prob = self.get_tag_proba(v, cur_cntx, norm)*pi[(k-1, w, u)] # max term
                    #     if w_prob > best_prob:
                    #         best_prob = w_prob
                    #         best_w = w

                    # best_idx = np.array(w_probs).argmax()
                    wuv_probs[(w, u, v)] = wuv_prob

                    # temp_pi[(k, u, v)] = best_prob # w_probs[best_idx]
                    # temp_bp[(k, u, v)] = best_w # S[k-2][best_idx]

            '''
            Beam-search selection:
            Select at each stage the B best tags for the sentence k'th word.
            Discard all others. Store the selected transitions S[k] = (u,v) for 3 best transitions. 
            '''

            # pi_items = temp_pi.items()
            # best_candidates = np.array([p for key, p in pi_items]).argsort()[::-1][:self.beam_search]
            best_candidates = np.array([p for key, p in wuv_probs.items()]).argsort()[::-1][:self.beam_search]
            best_transitions = []
            for idx, ((w,u,v), prob) in enumerate(wuv_probs.items()):
                if idx in best_candidates:
                    pi[(k, u, v)] = prob
                    bp[(k, u, v)] = w
                    best_transitions.append((u,v))
            S[k] = best_transitions

            # S[k] = [key[-1] for (idx, (key, val)) in enumerate(pi_items) if idx in best_candidates]
        # finish calculate pi, and bp.
        # start decoding:
        best_prob = 0
        y = [None] * N
        for (n, u, v), p in pi.items():
            if n == N-1:
                # probability of tagging word n-1 by u:
                w = bp[(n, u, v)]
                wbp = bp[(n-1,w,u)]
                u_cntx = Context.get_context_untagged(sentence, n-1, u, wbp, w)
                u_norm = self.get_norm_expmax((wbp, w, u), u_cntx)
                u_prob = self.get_tag_proba(u, u_cntx, norm=u_norm)
                v_cntx = Context.get_context_untagged(sentence, n, v, w, u)
                v_norm = self.get_norm_expmax((w, u, v), v_cntx)
                v_prob = self.get_tag_proba(v, v_cntx, norm=v_norm)
                total_prob = p * u_prob * v_prob
                if total_prob > best_prob:
                    y[-2] = u
                    y[-1] = v

        for k in range(N-2).__reversed__():
            y[k] = bp[(k+2, y[k+1], y[k+2])]

        infer_time = time.time() - t_start
        result_tags_str = [self.i2t[i] for i in y]
        return result_tags_str, infer_time








    def get_probs_for_ppt(self, ppt_candidates, context):
        '''
        Calculate softmax upon candidates for prev_prev_tag
        :param candidates_tags:
        :param context:
        :return:
        '''

    def get_norm_expmax(self, key, context):
        if key in self.softmax_dict.keys():
            return self.softmax_dict[key]
        else:
            norm, expmax = self.get_context_norm_and_expmax(context)
            self.softmax_dict[key] = (norm, expmax)
            return norm, expmax


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
        # print('optimizer iter no.', self.l_counter)
        # proba part, the linear term in l(v)
        # proba = 0
        # normalization part, i.e. sum_on_x(i){ log( sum_on_y'[exp(v*f(x(i),y'] ) }
        norm_part = 0
        for s_idx, sentence in enumerate(self.sentences):
            for w_idx in range(len(sentence)):
                # curr_context = Context.get_context_tagged(sentence, w_idx)
                dot_products = []
                for tag in self.tags:
                    # curr_context.tag = tag
                    # positive_features = self.get_positive_features_for_context(curr_context)
                    pf = self.training_features[s_idx][w_idx][tag]
                    dot_products.append(self.get_dot_product_from_positive_features(pf, v))

                safety_term = max(dot_products)
                exponents = np.exp(np.array(dot_products) - safety_term)
                # p = self.get_dot_product_from_positive_features(self.word_positive_indices[s_idx][w_idx], v) - safety_term # TODO optimize.
                # proba += p
                norm_part += np.log(sum(exponents)) + safety_term # we need to subtract finally the safety_term  that was in the nominator.
        proba = np.dot(self.empirical_counts, v)
        res = proba - norm_part - 0.5 * self.l2 * np.dot(v, v) # regularization l2
        print('L(v) iter np. ', self.l_counter, res)
        # self.log('l = ', self.l_counter, ',' ,res)
        self.l_counter += 1

        # return minus res because the optimizer does not know to maximize,
        # so we minimize (-l(v))
        return -res


    def grad_l(self, v):
        # self.log('grad = ', self.l_grad_counter)
        self.l_grad_counter += 1
        # expected counts
        expected_counts = np.zeros(v.shape)
        for s_idx, sentence in enumerate(self.sentences):
            for w_idx in range(len(sentence)):
                # curr_context = Context.get_context_tagged(sentence, w_idx)
                # f_xi_for_all_ytag = self.training_features[s_idx][w_idx] # the features f(x(i),y') for y' in Y
                dot_products = []  # f(x(i),y') @ v for y' in Y
                # safe softmax: first calculate all dot products, then deduce the max val to avoid overflow
                # for tag in self.tags:
                    # curr_context.tag = tag
                    # pf = f_xi_for_all_ytag[tag] #self.get_positive_features_for_context(curr_context)
                    # f_xi_for_all_ytag.append(f_xi_ytag)
                pf_per_tag = self.training_features[s_idx][w_idx]
                for pf in pf_per_tag:
                    dot_products.append(self.get_dot_product_from_positive_features(pf, v))
                dot_products = np.array(dot_products) - max(dot_products) # make the exponent safe.

                exponents = np.exp(dot_products)
                softmax_denominator = sum(exponents)
                softmax_xi_ytag = exponents / softmax_denominator

                # the nominator cannot be explicitely computed in positive feature representation.
                # the update is element-wise:
                if softmax_denominator > 0:
                    for tag, pos_f in enumerate(pf_per_tag):
                        for v_idx in pos_f:
                            expected_counts[v_idx] += softmax_xi_ytag[tag]


        res = self.empirical_counts - expected_counts - self.l2 * v # regularization l2
        print('dL(v) iter np. ', self.l_counter, res)
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
    print(datetime.datetime.now(), ' - loading data from file', filename)
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
        print('found ', len(bad_words),' bad words - ', bad_words)

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

    print(datetime.datetime.now(),' - found ', len(all_tags_set),' tags - ', all_tags_set)

    return sentences_int, w2i, int2word, tag2int, int2tag

def parse_test_set(filename, n=None):
    '''
    parse input file to sentences. convert words and tags to int according to the training corpus.
    :param      filename:
    :return:    sentences - parsed_sentences in integer format.
    '''
    sentences = []
    f_text = None
    print(datetime.datetime.now(),' - loading data from file ', filename)
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

def evaluate(model, testset_file, n_samples=1, max_words=None, version='avrech'):
    parsed_testset = parse_test_set(testset_file, n=n_samples) # TODO: use trainset dictionary for test set.
    predictions = []
    accuracy = []
    print(' ---------',version,'---------- ')
    for s_idx, sample in enumerate(parsed_testset):
        if max_words is None:
            nw = len(sample)
        else:
            nw = min(len(sample), max_words)
        sentence = ' '.join([word_id[0] for word_id in sample[:nw]])
        if version == 'avrech':
            results_tag, inference_time = model.infer_viterbi_beam_search(sentence)
        else:
            results_tag, inference_time = model.infer(sentence)

        predictions.append(results_tag)
        comparison = [results_tag[word_idx] == sample[word_idx][1] for word_idx in range(nw)]
        accuracy.append(sum(comparison) / len(comparison))
        tagged_sentence = ['{}_{}'.format(sentence.split(" ")[i], results_tag[i]) for i in range(len(results_tag))]
    print('results: time - ', "{0:.2f}".format(inference_time), '[sec], tags - ', " ".join(tagged_sentence))
    print('average accuracy: ', "{0:.2f}".format(np.mean(accuracy)))

if __name__ == "__main__":
    # load training set
    parsed_sentences, w2i, i2w, t2i, i2t = get_parsed_sentences_from_tagged_file('train.wtag', n=100)

    my_model = MEMM(parsed_sentences, w2i, i2w, t2i, i2t, beam_search=3)
    train_time = my_model.train_model()
    print('train: time - ', "{0:.2f}".format(train_time), '[sec]')
    with open('model_prm.pkl', 'wb') as f:
        pickle.dump(my_model.parameter_vector, f)
    # f = open('model_prm.pkl', 'rb')
    # param_vector = pickle.load(f)
    # model.test('bla', annotated=True)



    evaluate(my_model, 'train.wtag', n_samples=1, version='old')

    # Evaluate test set:
    evaluate(my_model, 'train.wtag', n_samples=1, version='avrech')
