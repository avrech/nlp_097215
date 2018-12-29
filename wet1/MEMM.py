# from collections import Counter
import numpy as np
import math
import datetime
import scipy
from scipy import optimize
import pickle
import time

from tqdm import tqdm


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
        history = [w for w in sentence[:index]]
        prev_tag = sentence[index - 1][1] if index > 0 else None
        prev_prev_tag = sentence[index - 2][1] if index > 1 else None
        next_word = sentence[index + 1][0] if len(sentence) > index + 1 else None
        return Context(word, tag, history, prev_tag, prev_prev_tag, next_word)

    @staticmethod
    def get_context_untagged(sentence, index, tag, prev_prev_tag, prev_tag):
        """Creates a context from an untagged sentence"""
        word = sentence[index]
        history = [w for w in sentence[:index]]
        next_word = sentence[index + 1] if len(sentence) > index + 1 else None
        return Context(word, tag, history, prev_tag, prev_prev_tag, next_word)


class MEMM:
    WORD_UNIGRAMS = 'WORD_UNIGRAMS'
    WORD_BIGRAMS = 'WORD_BIGRAMS'
    WORD_TRIGRAMS = 'WORD_TRIGRAMS'
    TAG_UNIGRAMS = 'TAG_UNIGRAMS'
    TAG_BIGRAMS = 'TAG_BIGRAMS'
    TAG_TRIGRAMS = 'TAG_TRIGRAMS'
    WORD_TAG_PAIRS = 'WORD_TAG_PAIRS'
    PREFIX_TAG = 'PREFIX_TAG'
    SUFFIX_TAG = 'SUFFIX_TAG'
    CAPITAL_FIRST_LETTER_TAG = 'CAPITAL_FIRST_LETTER_TAG'
    CAPITAL_WORD_TAG = 'CAPITAL_WORD_TAG'
    NUMBER_TAG = 'NUMBER_TAG'
    FIRST_WORD_TAG = 'FIRST_WORD_TAG'
    SECOND_WORD_TAG = 'SECOND_WORD_TAG'
    LAST_WORD_TAG = 'LAST_WORD_TAG'
    PREV_WORD_CUR_TAG = 'PREV_WORD_CURR_TAG'
    NEXT_WORD_CUR_TAG = 'NEXT_WORD_CURR_TAG'

    def __init__(self, model_params):
        self.tags = []
        self.enriched_tags = []
        self.feature_set = {}
        self.num_features = 0
        self.suffix_threshold = {}
        self.prefix_threshold = {}
        self.word_vectors = None
        self.sentences = None
        self.parameter_vector = None
        self.empirical_counts = None
        self.word_positive_indices = None
        self.l_counter = self.l_grad_counter = 0
        self.safe_softmax = True
        self.verbose = 1
        self.last_known_v = -math.inf
        self.num_interations = 0
        self.train_start = None
        self.train_end = None
        self.test_start = None
        self.test_end = None
        self.params = model_params
        self.beam_min = int(self.params['beam_min'])

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
                next_word = sentence[i + 1][0] if i < len(sentence) - 1 else None

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

                if prev_word:
                    self.safe_add(prev_word_cur_tag, (prev_word, tag))
                if next_word:
                    self.safe_add(next_word_cur_tag, (next_word, tag))

        return {self.WORD_UNIGRAMS: word_unigrams_counts,
                self.TAG_UNIGRAMS: tag_unigrams_counts,
                self.WORD_TAG_PAIRS: word_tag_counts,
                self.WORD_BIGRAMS: word_bigrams_counts,
                self.TAG_BIGRAMS: tag_bigrams_counts,
                self.WORD_TRIGRAMS: word_trigrams_counts,
                self.TAG_TRIGRAMS: tag_trigrams_counts,
                self.PREFIX_TAG: prefix_tag_counter,
                self.SUFFIX_TAG: suffix_tag_counter,
                self.CAPITAL_FIRST_LETTER_TAG: capital_first_letter_tag,
                self.CAPITAL_WORD_TAG: capital_word_tag,
                self.NUMBER_TAG: number_tag,
                self.FIRST_WORD_TAG: first_word_tag,
                self.SECOND_WORD_TAG: second_word_tag,
                self.LAST_WORD_TAG: last_word_tag,
                self.PREV_WORD_CUR_TAG: prev_word_cur_tag,
                self.NEXT_WORD_CUR_TAG: next_word_cur_tag,
                }

    def add_features_subset(self, features_dict, dict_name, start_index):
        index_list = list(features_dict[dict_name].keys())
        self.feature_set[dict_name] = {key: index_list.index(key) + start_index for key in index_list}
        start_index += len(index_list)
        return start_index

    def train_model(self, sentences):
        self.train_start = datetime.datetime.now()
        # prepare data and get statistics
        print('{} - processing data'.format(datetime.datetime.now()))
        self.sentences = sentences
        text_stats = self.get_text_statistics()

        # filtering small affixes (by taking only top 10% of each affix length)
        affix_percent = int(self.params['affix_precent'])
        for i in range(4):
            curr_threshold = math.floor(np.percentile([count for suf, count in text_stats[self.SUFFIX_TAG].items()
                                                       if len(suf[0]) == i + 1], affix_percent))
            self.suffix_threshold[i + 1] = curr_threshold
            curr_threshold = math.floor(np.percentile([count for suf, count in text_stats[self.PREFIX_TAG].items()
                                                       if len(suf[0]) == i + 1], affix_percent))
            self.prefix_threshold[i + 1] = curr_threshold
        # self.suff_th = {1: 200, 2: 100, 3: 50, 4: 8}
        text_stats[self.SUFFIX_TAG] = {(s, t): count_st for (s, t), count_st in text_stats[self.SUFFIX_TAG].items() if
                                       count_st > self.suffix_threshold[len(s)]}
        text_stats[self.PREFIX_TAG] = {(p, t): count_pt for (p, t), count_pt in text_stats[self.PREFIX_TAG].items() if
                                       count_pt > self.prefix_threshold[len(p)]}

        print('{} - preparing features'.format(datetime.datetime.now()))
        self.tags = list(text_stats[self.TAG_UNIGRAMS].keys())
        self.enriched_tags = [None] + self.tags

        # define the features set
        start_index = 0
        # feature-word pairs in dataset (#100)
        start_index = self.add_features_subset(text_stats, self.WORD_TAG_PAIRS, start_index)
        # suffixes <= 4 and tag pairs in dataset (#101)
        start_index = self.add_features_subset(text_stats, self.SUFFIX_TAG, start_index)
        # prefixes <= 4 and tag pairs in dataset (#102)
        start_index = self.add_features_subset(text_stats, self.PREFIX_TAG, start_index)
        # tag trigrams in datset (#103)
        start_index = self.add_features_subset(text_stats, self.TAG_TRIGRAMS, start_index)
        # tag bigrams in datset (#104)
        start_index = self.add_features_subset(text_stats, self.TAG_BIGRAMS, start_index)
        # tag unigrams in datset (#105)
        start_index = self.add_features_subset(text_stats, self.TAG_UNIGRAMS, start_index)
        # capital first letter tag
        start_index = self.add_features_subset(text_stats, self.CAPITAL_FIRST_LETTER_TAG, start_index)
        # capital word tag
        start_index = self.add_features_subset(text_stats, self.CAPITAL_WORD_TAG, start_index)
        # number tag feature
        start_index = self.add_features_subset(text_stats, self.NUMBER_TAG, start_index)
        # first word in sentence tag
        start_index = self.add_features_subset(text_stats, self.FIRST_WORD_TAG, start_index)
        # second word in sentence tag
        start_index = self.add_features_subset(text_stats, self.SECOND_WORD_TAG, start_index)
        # previous word + current tag pairs (#106)
        start_index = self.add_features_subset(text_stats, self.PREV_WORD_CUR_TAG, start_index)
        # next word + current tag pairs (#107)
        start_index = self.add_features_subset(text_stats, self.NEXT_WORD_CUR_TAG, start_index)
        self.num_features = start_index

        # print feature sizes
        for dict_name in text_stats:
            print(f'features subset: {dict_name}, num of features: {len(text_stats[dict_name])}')
        print(f'A total of {self.num_features} features')

        # for each word in the corpus, find its feature vector (only positive indices)
        print(f'{datetime.datetime.now()} - computing features for train data')
        word_positive_indices = {}
        contexts_dict = {}
        for sentence in sentences:
            for i in range(len(sentence)):
                context = Context.get_context_tagged(sentence, i)
                positive_indices = self.get_positive_features_for_context(context)
                word_positive_indices[(sentence, i)] = positive_indices
                contexts_dict[(sentence, i)] = context
        self.word_positive_indices = word_positive_indices
        self.empirical_counts = self.get_empirical_counts_from_dict()

        # calculate the parameters vector
        if self.params['from_pickle']:
            print(f'{datetime.datetime.now()} - loading initial parameter vector')
            input_param_vec_f = open(self.params['pickle_input'], 'rb')
            x0 = pickle.load(input_param_vec_f)
        else:
            x0 = np.ones(self.num_features)

        maxiter = int(self.params['maxiter'])
        if maxiter > 0:
            print(f'{datetime.datetime.now()} - finding parameter vector')
            param_vector = scipy.optimize.minimize(fun=self.ml, x0=x0, method='L-BFGS-B', jac=self.grad_l,
                                                   options={'disp': True, 'maxiter': int(self.params['maxiter'])})
            self.parameter_vector = param_vector.x
            with open(self.params['pickle_output'], 'wb') as f:
                pickle.dump(self.parameter_vector, f)
        else:
            # call ml just to calculate L(v)
            self.ml(x0)
            self.parameter_vector = x0

        print(f'{datetime.datetime.now()} - model train complete')
        self.train_end = datetime.datetime.now()
        return

    # use Viterbi to infer tags for the target sentence
    def infer(self, sentence):
        # print(f'{datetime.datetime.now()} - predict for:')
        # print(sentence)
        t_start = time.time()
        parsed_sentence = tuple(word.rstrip() for word in sentence.split(' '))
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
            # find the probabilty above which we have at least beam_min possibilities
            min_proba_for_stage = 1
            count_entries_per_proba = len([pi_val for pi_key, pi_val in pi_temp.items()
                                           if pi_val >= min_proba_for_stage])

            while count_entries_per_proba < self.beam_min:
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
        if len(parsed_sentence) == 1:
            result_tags = result_tags[-1:]
        else:
            for word_index in range(len(parsed_sentence) - 1, 1, -1):
                index, tn_prev, tn, proba = [x for x in sorted_pi if x[0] == word_index and x[2] == result_tags[0]][0]
                result_tags = [tn_prev] + result_tags
        infer_time = time.time() - t_start
        return result_tags, infer_time

    def get_positive_features_for_context(self, context):
        positive_indices = [self.feature_set[self.WORD_TAG_PAIRS].get((context.word, context.tag), None)]
        for i in range(min(4, len(context.word) - 1)):
            positive_indices.append(self.feature_set[self.SUFFIX_TAG].get((context.word[-i-1:], context.tag), None))
        for i in range(min(4, len(context.word))):
            positive_indices.append(self.feature_set[self.PREFIX_TAG].get((context.word[:i+1], context.tag), None))
        positive_indices.append(
            self.feature_set[self.TAG_TRIGRAMS].get((context.prev_prev_tag, context.prev_tag, context.tag), None))
        positive_indices.append(self.feature_set[self.TAG_BIGRAMS].get((context.prev_tag, context.tag), None))
        positive_indices.append(self.feature_set[self.TAG_UNIGRAMS].get(context.tag, None))

        if context.word[0].isupper():
            positive_indices.append(self.feature_set[self.CAPITAL_FIRST_LETTER_TAG].get(context.tag, None))
        if all([letter.isupper() for letter in context.word]):
            positive_indices.append(self.feature_set[self.CAPITAL_WORD_TAG].get(context.tag, None))
        if context.word.replace('.', '', 1).isdigit():
            positive_indices.append(self.feature_set[self.NUMBER_TAG].get(context.tag, None))

        if context.index == 0:
            positive_indices.append(self.feature_set[self.FIRST_WORD_TAG].get(context.tag, None))
        else:
            positive_indices.append(self.feature_set[self.PREV_WORD_CUR_TAG].get((context.history[-1], context.tag),
                                                                                 None))
        if context.index == 1:
            positive_indices.append(self.feature_set[self.SECOND_WORD_TAG].get(context.tag, None))

        if context.next_word:
            positive_indices.append(self.feature_set[self.NEXT_WORD_CUR_TAG].get((context.next_word, context.tag),
                                                                                 None))

        positive_indices = [x for x in positive_indices if x]
        return positive_indices

    def get_empirical_counts_from_dict(self):
        emprical_counts = np.zeros(self.num_features)
        for positive_features in self.word_positive_indices.values():
            emprical_counts[positive_features] += 1
        return emprical_counts

    @staticmethod
    def get_dot_product_from_positive_features(positive_indices, v):
        dot_product = sum(v[positive_indices])
        return dot_product

    # soft max
    def get_tag_proba(self, tag, context, norm=None):
        context.tag = tag
        positive_features = self.get_positive_features_for_context(context)
        dot_product = self.get_dot_product_from_positive_features(positive_features, self.parameter_vector)
        numerator = np.exp(dot_product)
        if norm is None:
            norm = self.get_context_norm(context)
        proba = numerator / norm if norm > 0 else 0
        return proba

    def get_context_norm(self, context):
        norm = 0
        for curr_tag in self.tags:
            context.tag = curr_tag
            positive_features = self.get_positive_features_for_context(context)
            dot_product = self.get_dot_product_from_positive_features(positive_features, self.parameter_vector)
            norm += np.exp(dot_product)
        return norm

    def log(self, msg):
        if self.verbose:
            print(msg)

    # the ML estimate maximization function
    def ml(self, v):
        # proba part
        proba = 0
        # normalization part
        norm_part = 0
        for sentence in tqdm(self.sentences, 'l sentences'):
            for i in range(len(sentence)):
                proba += self.get_dot_product_from_positive_features(self.word_positive_indices[(sentence, i)], v)
                curr_context = Context.get_context_tagged(sentence, i)
                curr_exp = 0

                for tag in self.tags:
                    curr_context.tag = tag
                    positive_features = self.get_positive_features_for_context(curr_context)
                    curr_exp += np.exp(self.get_dot_product_from_positive_features(positive_features, v))
                norm_part += np.log(curr_exp)

        self.last_known_v = proba - norm_part
        self.log(f'{datetime.datetime.now()} - l = {self.l_counter},{self.last_known_v}')
        self.l_counter += 1
        return -self.last_known_v

    def grad_l(self, v):
        # expected counts
        expected_counts = 0
        for sentence in tqdm(self.sentences, 'grad l sentences'):
            for i in range(len(sentence)):
                curr_context = Context.get_context_tagged(sentence, i)
                normalization = 0
                numerator = np.zeros(self.num_features)
                dot_products = []
                vectors = []
                # safe softmax: first calculate all dot products, then deduce the max val to avoid overflow
                for tag in self.tags:
                    curr_context.tag = tag
                    curr_positive_features = self.get_positive_features_for_context(curr_context)
                    vectors.append(curr_positive_features)
                    dot_products.append(self.get_dot_product_from_positive_features(curr_positive_features, v))
                dot_products = np.array(dot_products) - max(dot_products)
                for j, product in enumerate(dot_products):
                    curr_exp = np.exp(product)
                    normalization += curr_exp
                    numerator[vectors[j]] += curr_exp

                expected_counts += numerator / normalization if normalization > 0 else 0
        res = self.empirical_counts - expected_counts
        self.log(f'{datetime.datetime.now()} - grad = {self.l_grad_counter}')
        self.l_grad_counter += 1
        return -res


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
        words_and_tags = tuple([tuple(tagged_word.split('_')) for tagged_word in tagged_words])
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


def tag_file(model, filename):
    input_f = open(filename)
    predicted_sentences = []
    for sentence in tqdm(input_f, 'tagging sentences...'):
        sentence = sentence.rstrip()
        split_sentence = sentence.split(' ')
        results_tag, inference_time = model.infer(sentence)
        tagged_sentence = ' '.join(['{}_{}'.format(split_sentence[i], results_tag[i]) for i in range(len(results_tag))])
        predicted_sentences.append(tagged_sentence)
        with open(filename.split('.')[0] + '_m2_043493386.wtag', 'a+') as out_f:
            out_f.write(tagged_sentence + '\n')


def evaluate(model, testset_file, n_samples=None):
    if n_samples == 0:
        return 0
    parsed_testset = get_parsed_sentences_from_tagged_file(testset_file)
    predictions = []
    sentence_accuracy = []
    predicted_sentences = []
    correct_predictions = 0
    total_predictions = 0
    conf_matrix = {true_tag: {predicted_tag: 0 for predicted_tag in model.tags} for true_tag in model.tags}
    for sentence in tqdm(parsed_testset[:n_samples], 'inferring sentences...'):
        unparsed_sentence = ' '.join([word[0] for word in sentence])
        results_tag, inference_time = model.infer(unparsed_sentence)
        predictions.append(results_tag)
        for i, predicted_tag in enumerate(results_tag):
            true_tag = sentence[i][1]
            if true_tag not in model.tags or predicted_tag not in model.tags:
                print('Error: unknown tag in test data')
                continue
            conf_matrix[true_tag][predicted_tag] += 1
        comparison = [results_tag[word_idx] == sentence[word_idx][1] for word_idx in range(len(sentence))]
        correct_predictions += sum(comparison)
        total_predictions += len(comparison)
        sentence_accuracy.append(sum(comparison) / len(comparison))
        tagged_sentence = ' '.join(['{}_{}'.format(sentence[i][0], results_tag[i]) for i in range(len(results_tag))])
        predicted_sentences.append(tagged_sentence)
        with open(testset_file.split('.')[0] + '_output.txt', 'a+') as out_f:
            out_f.write(tagged_sentence + '\n')

    accuracy = np.mean(sentence_accuracy)
    # flatten confusion matrix, find top errors
    flat_conf_mat = []
    for true_tag in conf_matrix:
        for predicted_tag in conf_matrix[true_tag]:
            flat_conf_mat.append((true_tag, predicted_tag, conf_matrix[true_tag][predicted_tag]))
    sorted_flat_conf_mat = sorted(flat_conf_mat, key=lambda x: x[2], reverse=True)
    true_tags_top_errors = set()
    pred_tags_top_errors = set()
    curr_row = 0
    while len(true_tags_top_errors) < 10 and curr_row < len(sorted_flat_conf_mat):
        curr_true_tag = sorted_flat_conf_mat[curr_row][0]
        curr_pred_tag = sorted_flat_conf_mat[curr_row][1]
        curr_row += 1
        if curr_true_tag == curr_pred_tag:
            continue
        true_tags_top_errors.add(curr_true_tag)
        pred_tags_top_errors.add(curr_pred_tag)
    tags_top_errors = list(pred_tags_top_errors.union(true_tags_top_errors))
    print('top errors confusion matrix:')
    print(f'true tag,{",".join(tags_top_errors)}')
    for true_tag in tags_top_errors:
        err_str = ','.join([str(conf_matrix[true_tag][pred_tag]) for pred_tag in tags_top_errors])
        print(f'{true_tag},{err_str}')
    print('full confusion matrix:')
    print(f'true tag,{",".join(model.tags)}')
    for true_tag in tags_top_errors:
        err_str = ','.join([str(conf_matrix[true_tag][pred_tag]) for pred_tag in model.tags])
        print(f'{true_tag},{err_str}')
    print(f'average accuracy: {"{0:.2f}".format(accuracy)}')
    return correct_predictions / total_predictions


def compare_files(tagged_file, predicted_file, tags_list):
    tagged_sentences = get_parsed_sentences_from_tagged_file(tagged_file)
    pred_sentences = get_parsed_sentences_from_tagged_file(predicted_file)
    correct_predictions = 0
    total_predictions = 0
    conf_matrix = {true_tag: {predicted_tag: 0 for predicted_tag in tags_list} for true_tag in tags_list}
    for sentence_index, tagged_sentence in tqdm(enumerate(tagged_sentences), 'comparing sentences...'):
        pred_sentence = pred_sentences[sentence_index]
        if len(tagged_sentence) != len(pred_sentence):
            print(f'Error in sentence {sentence_index}: different lengths')
            continue
        different_words = [tagged_sentence[word_idx][0] != pred_sentence[word_idx][0]
                           for word_idx in range(len(tagged_sentence))]
        if sum(different_words):
            print(f'Error in sentence {sentence_index}: different word')
            continue
        tag_comparison = [tagged_sentence[word_idx][1] == pred_sentence[word_idx][1]
                           for word_idx in range(len(tagged_sentence))]
        correct_predictions += sum(tag_comparison)
        total_predictions += len(tag_comparison)
        for i, tag_word in enumerate(tagged_sentence):
            true_tag = tag_word[1]
            pred_tag = pred_sentence[i][1]
            if true_tag not in tags_list or pred_tag not in tags_list:
                print(f'Error in sentence {sentence_index}: unknown tag in test data')
                continue
            conf_matrix[true_tag][pred_tag] += 1

    accuracy = correct_predictions / total_predictions
    # flatten confusion matrix, find top errors
    flat_conf_mat = []
    for true_tag in conf_matrix:
        for predicted_tag in conf_matrix[true_tag]:
            flat_conf_mat.append((true_tag, predicted_tag, conf_matrix[true_tag][predicted_tag]))
    sorted_flat_conf_mat = sorted(flat_conf_mat, key=lambda x: x[2], reverse=True)
    true_tags_top_errors = set()
    pred_tags_top_errors = set()
    curr_row = 0
    while len(true_tags_top_errors) < 10 and curr_row < len(sorted_flat_conf_mat):
        curr_true_tag = sorted_flat_conf_mat[curr_row][0]
        curr_pred_tag = sorted_flat_conf_mat[curr_row][1]
        curr_row += 1
        if curr_true_tag == curr_pred_tag:
            continue
        true_tags_top_errors.add(curr_true_tag)
        pred_tags_top_errors.add(curr_pred_tag)
    tags_top_errors = list(pred_tags_top_errors.union(true_tags_top_errors))
    print('top errors confusion matrix:')
    print(f'true tag,{",".join(tags_top_errors)}')
    for true_tag in tags_top_errors:
        err_str = ','.join([str(conf_matrix[true_tag][pred_tag]) for pred_tag in tags_top_errors])
        print(f'{true_tag},{err_str}')
    print('full confusion matrix:')
    print(f'true tag,{",".join(tags_list)}')
    for true_tag in tags_top_errors:
        err_str = ','.join([str(conf_matrix[true_tag][pred_tag]) for pred_tag in tags_list])
        print(f'{true_tag},{err_str}')
    print(f'average accuracy: {"{0:.3f}".format(accuracy)}')
    return correct_predictions / total_predictions

if __name__ == "__main__":
    pass
