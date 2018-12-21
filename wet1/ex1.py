# from collections import Counter
import numpy as np
import math
import datetime
import scipy
from scipy import optimize
import pickle
import time
# receives input file (of tagged sentences),
# returns a list of sentences, each sentence is a list of (word,tag)s
# also performs base verification of input
def get_parsed_sentences_from_tagged_file(filename):
    print(datetime.datetime.now(),' - loading data from file {filename}')
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
        print('found ',len(bad_words),' bad words - ', bad_words)
    all_tags_list = [word[1] for word in all_words_and_tags]
    all_tags_set = set(all_tags_list)
    # tags_counter = Counter(all_tags_list)
    # all_words_list = [word[0] for word in all_words_and_tags]
    # words_counter = Counter(all_words_list)
    # all_words_set = set(all_words_list)
    print(datetime.datetime.now(),' - found ',len(all_tags_set),' tags - ',all_tags_set)
    return sentences

def pre_process_data(train_set):
    # load training set
    parsed_sentences = get_parsed_sentences_from_tagged_file('train.wtag')
    words = [word_tag_pair[0] for sentence in parsed_sentences for word_tag_pair in sentence]

    # Extract prefix:
    prefix = {1: {}, 2: {}, 3: {}, 4: {}}
    suffix = {1: {}, 2: {}, 3: {}, 4: {}}
    for word in words:
        for l in np.arange(1, 5):
            if word.__len__() >= l:
                pref = word[:l]
                prefix[l][pref] = prefix[l].get(pref, 0) + 1
                suff = word[-l:]
                suffix[l][pref] = suffix[l].get(suff, 0) + 1
    # Define thresholds:
    pref_th = {1: 1000, 2: 1000, 3: 500, 4: 500}
    selected_prefix = {1: [], 2: [], 3: [], 4: []}
    # print('prefix counts:')
    for l in np.arange(1, 5):
        for p in prefix[l].keys():
            if prefix[l][p] > pref_th[l]:
                selected_prefix[l].append(p)
                # print(p, ': ', prefix[l][p])

    suff_th = {1: 100, 2: 100, 3: 100, 4: 100}
    selected_suffix = {1: [], 2: [], 3: [], 4: []}
    # print('suffix counts:')
    for l in np.arange(1, 5):
        for p in suffix[l].keys():
            if suffix[l][p] > suff_th[l]:
                selected_suffix[l].append(p)
                # print(p, ': ', suffix[l][p])

    print('# prefix features: ',
          '-1', selected_prefix[1].__len__(),
          '-2', selected_prefix[2].__len__(),
          '-3', selected_prefix[3].__len__(),
          '-4', selected_prefix[4].__len__(),
          ' | Total-', sum([p.__len__() for p in selected_prefix.values()]))
    print('# suffix features: ',
          '-1', selected_suffix[1].__len__(),
          '-2', selected_suffix[2].__len__(),
          '-3', selected_suffix[3].__len__(),
          '-4', selected_suffix[4].__len__(),
          ' | Total-', sum([p.__len__() for p in selected_suffix.values()]))

    print('preprocess finished')
    return parsed_sentences, selected_prefix, selected_suffix

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
    BEAM_MIN = 10

    def __init__(self):
        self.tags = []
        self.enriched_tags = []
        self.feature_set = []
        self.feature_set1 = []
        self.word_vectors = None
        self.sentences = None
        self.parameter_vector = None
        self.empirical_counts = None
        self.word_positive_indices = None
        self.use_vector_form = False

    def train_model(self, sentences, prefix=None, suffix=None, param_vec=None):
        t_start = time.time()
        # prepare data and get statistics
        print(datetime.datetime.now(), ' - processing data')
        self.sentences = sentences
        word_tag_pairs = []
        # word_number = sum([len(sentence) for sentence in sentences])
        for sentence in sentences:
            for word_tag in sentence:
                if word_tag not in word_tag_pairs:
                    word_tag_pairs.append(word_tag)
                if word_tag[1] not in self.tags:
                    self.tags.append(word_tag[1])

        self.enriched_tags = [None] + self.tags

        # define the features set:
        # word-tag pairs in dataset
        self.feature_set += [(lambda w, t:(lambda cntx: 1 if cntx.word == w and cntx.tag == t else 0))(w, t)
                             for w, t in word_tag_pairs]
        # prefixes <= 4 and tag pairs in dataset
        if prefix is not None:
            self.feature_set += [(lambda pref, t:(lambda cntx: 1 if cntx.word[:pref.__len__()] == pref and cntx.tag == t else 0))(pref, t)
                                 for preflist in prefix.values() for pref in preflist for t in self.tags]
        # suffixes <= 4 and tag pairs in dataset
        if suffix is not None:
            self.feature_set += [(lambda pref, t:(lambda cntx: 1 if cntx.word[:pref.__len__()] == pref and cntx.tag == t else 0))(suff, t)
                                 for sufflist in suffix.values() for suff in sufflist for t in self.tags]
        # tag trigrams in datset

        # tag bigrams in datset
        # tag unigrams in datset
        # previous word + current tag pairs
        # next word + current tag pairs

        # for each word in the corpus, find its feature vector
        word_vectors = []
        word_positive_indices = {}
        for sentence in sentences:
            for i in range(len(sentence)):
                context = Context.get_context_tagged(sentence, i)
                if self.use_vector_form:
                    vector = self.get_feature_vector_for_context(context)
                    word_vectors.append(vector)
                else:
                    positive_indices = self.get_positive_features_for_context(context)
                    word_positive_indices[(sentence, i)] = positive_indices
        if self.use_vector_form:
            self.word_vectors = np.array(word_vectors)
            self.empirical_counts = np.sum(self.word_vectors, axis=0)
        else:
            self.word_positive_indices = word_positive_indices
            self.empirical_counts = self.get_empirical_counts_from_dict()

        # calculate the parameters vector
        print(datetime.datetime.now(), ' - finding parameter vector')
        if param_vec is None:
            if self.use_vector_form:
                param_vec = scipy.optimize.minimize(fun=self.l_vector, x0=np.ones(len(self.feature_set)),
                                                    method='L-BFGS-B', jac=self.grad_l_vector)
            else:
                param_vec = scipy.optimize.minimize(fun=self.l, x0=np.ones(len(self.feature_set)), method='L-BFGS-B',
                                                    jac=self.grad_l)
        # param_vec = scipy.optimize.fmin_l_bfgs_b(self.l, np.zeros(len(self.feature_set)), self.grad_l)
        # optimize.minimize(self.l,np.zeros(len(self.feature_set)),)
        self.parameter_vector = param_vec.x
        print(self.parameter_vector)
        print(datetime.datetime.now(), ' - model train complete')
        return time.time() - t_start

    # use Viterbi to infer tags for the target sentence
    def infer(self, sentence):
        print(datetime.datetime.now() ,' - predict for {sentence}')
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
        return np.nonzero(np.array([feature(context) for feature in self.feature_set]))

    def get_empirical_counts_from_dict(self):
        emprical_counts = np.zeros(len(self.feature_set))
        for positive_features in self.word_positive_indices.values():
            emprical_counts[positive_features] += 1
        return emprical_counts

    def get_dot_product(self, feature_vector):
        dot_product = sum([feature_vector[i] * self.parameter_vector[i] for i in range(len(feature_vector))])
        return dot_product

    @staticmethod
    def get_dot_product_from_positive_features(positive_indices, v):
        return np.sum(np.take(v, positive_indices))

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
        proba = 0
        # normalization part
        norm_part = 0
        for sentence in self.sentences:
            for i in range(len(sentence)):
                proba += self.get_dot_product_from_positive_features(self.word_positive_indices[(sentence, i)], v)
                curr_context = Context.get_context_tagged(sentence, i)
                curr_exp = 0
                for tag in self.tags:
                    curr_context.tag = tag
                    # vector = self.get_feature_vector_for_context(curr_context)
                    positive_features = self.get_positive_features_for_context(curr_context)
                    curr_exp += 2 ** self.get_dot_product_from_positive_features(positive_features, v)
                norm_part += math.log(curr_exp, 2)
        return -(proba - norm_part)

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
        return -(proba - norm_part)

    def grad_l(self, v):
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
                    curr_positive_features = self.word_positive_indices[(sentence, i)]
                    curr_exp = 2 ** self.get_dot_product_from_positive_features(curr_positive_features, v)
                    normalization += curr_exp
                    nominator += np.multiply(vector, curr_exp)

                expected_counts += nominator / normalization if normalization > 0 else 0
        return -(self.empirical_counts - expected_counts)

    def grad_l_vector(self, v):
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
        return -(self.empirical_counts - expected_counts)

    def test(self, text, annotated=False):
        pass


# receives input file (of tagged sentences),
# returns a list of sentences, each sentence is a list of (word,tag)s
# also performs base verification of input
def get_parsed_sentences_from_tagged_file(filename):
    print(datetime.datetime.now(), ' - loading data from file {filename}')
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
        print('found ', len(bad_words),' bad words - ',bad_words)
    all_tags_list = [word[1] for word in all_words_and_tags]
    all_tags_set = set(all_tags_list)
    # tags_counter = Counter(all_tags_list)
    # all_words_list = [word[0] for word in all_words_and_tags]
    # words_counter = Counter(all_words_list)
    # all_words_set = set(all_words_list)
    print(datetime.datetime.now(), ' - found ', len(all_tags_set), ' tags - ', all_tags_set)
    return sentences

def evaluate(model, testset_file, n_samples, max_words=None):
    parsed_testset = get_parsed_sentences_from_tagged_file(testset_file)
    predictions = []
    accuracy = []
    for ii, sample in enumerate(parsed_testset[:n_samples]):
        sentence = ' '.join([word[0] for word in sample[:max_words]])
        results_tag, inference_time = model.infer(sentence)
        predictions.append(results_tag)
        comparison = [results_tag[word_idx] == sample[word_idx][1] for word_idx in range(max_words)]
        accuracy.append(sum(comparison) / len(comparison))
        tagged_sentence = [f'{sentence.split(" ")[i]}_{results_tag[i]}' for i in range(len(results_tag))]
        print('results: time - ', "{0:.2f}".format(inference_time),'[sec], tags - ', " ".join(tagged_sentence))
    print('average accuracy: ', "{0:.2f}".format(np.mean(accuracy)))

if __name__ == "__main__":
    # load training set
    # preprocess data, extract relevant prefix/suffix etc.
    parsed_sentences, pref, suff = pre_process_data('train.wtag')

    model = MEMM()
    train_time = model.train_model(parsed_sentences[:1], prefix=pref, suffix=suff)
    print('train: time - ', "{0:.2f}".format(train_time),'[sec]')
    with open('model_prm.pkl', 'wb') as f:
        pickle.dump(model.parameter_vector, f)
    # f = open('model_prm.pkl', 'rb')
    # param_vector = pickle.load(f)
    # model.test('bla', annotated=True)
    sentence1 = ' '.join([word[0] for word in parsed_sentences[0]])
    results_tag, inference_time = model.infer(sentence1)
    tagged_sentence1 = ['{}_{}'.format(sentence1.split(" ")[i], results_tag[i]) for i in range(len(results_tag))]
    # print(f'results({inference_time}[sec]: {" ".join(tagged_sentence1)}')
    print('results: time - ', "{0:.2f}".format(inference_time),'[sec], tags - '," ".join(tagged_sentence1))

    # Evaluate test set:
    evaluate(model, 'test.wtag', 1, 5)
