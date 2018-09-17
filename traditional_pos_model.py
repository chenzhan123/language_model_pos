import nltk
import random
from collections import defaultdict
import pickle
from nltk.tag.api import TaggerI
from nltk.data import find, load

PICKLE = "averaged_perceptron_tagger.pickle"
POS_TAG=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','LS','MD','NN','NNS','NNP','NNPS','PDT',
         'POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBZ','VBP','VBD','VBN','VBG','WDT',
         'WP','WP$','WRB']

def load_universal_map(path):
    contents = load(path, format="text")
    _MAPPINGS = defaultdict()
    for line in contents.splitlines():
        line = line.strip()
        if line == '':
            continue
        fine, coarse = line.split('\t')
        _MAPPINGS[fine] = coarse
    return _MAPPINGS

class AveragedPerceptron(object):

    '''An averaged perceptron, as implemented by Matthew Honnibal.

    See more implementation details here:
        http://spacy.io/blog/part-of-speech-POS-tagger-in-python/
    '''

    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        self.classes = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features):
        '''Dot-product the features and current weights and return the best label.'''
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
        # Do a secondary alphabetic sort, for stability
        left=max(self.classes, key=lambda label: (scores[label], label))
        # right=scores[left]
        # return [left,right]
        return max(self.classes, key=lambda label: (scores[label], label))

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / self.i, 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights

    def save(self, path):
        '''Save the pickled model weights.'''
        with open(path, 'wb') as fout:
            return pickle.dump(dict(self.weights), fout)

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = load(path)

class PerceptronTagger(TaggerI):
    '''
    Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://spacy.io/blog/part-of-speech-POS-tagger-in-python/

    '''

    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']

    def __init__(self, load=True):
        '''
        :param load: Load the pickled model upon instantiation.
        '''
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            AP_MODEL_LOC = 'file:' + str(find('taggers/averaged_perceptron_tagger/' + PICKLE))
            self.load(AP_MODEL_LOC)

    def tag(self, tokens):
        '''
        Tag tokenized sentences.
        :params tokens: list of word
        :type tokens: list(str)
        '''
        prev, prev2 = self.START
        output = []

        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            tag = self.tagdict.get(word)
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                # tag = self.model.predict(features)
                scores = defaultdict(float)
                for feat, value in features.items():
                    if feat not in self.model.weights or value == 0:
                        continue
                    weights = self.model.weights[feat]
                    for label, weight in weights.items():
                        scores[label] += value * weight
                # Do a secondary alphabetic sort, for stability
                left = max(self.classes, key=lambda label: (scores[label], label))
                right = scores[left]
                tag=[left, right]

            if isinstance(tag,str):
                output.append([word, tag])
            else:
                output.append([word, tag[0], tag[1]])
            prev2 = prev
            prev = tag[0]

        return output

    def train(self, sentences, save_loc=None, nr_iter=5):
        '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.

        :param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''
        # We'd like to allow ``sentences`` to be either a list or an iterator,
        # the latter being especially important for a large training dataset.
        # Because ``self._make_tagdict(sentences)`` runs regardless, we make
        # it populate ``self._sentences`` (a list) with all the sentences.
        # This saves the overheard of just iterating through ``sentences`` to
        # get the list by ``sentences = list(sentences)``.

        self._sentences = list()  # to be populated by self._make_tagdict...
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for sentence in self._sentences:
                words, tags = zip(*sentence)

                prev, prev2 = self.START
                context = self.START + [self.normalize(w) for w in words] \
                          + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            random.shuffle(self._sentences)

        # We don't need the training sentences anymore, and we don't want to
        # waste space on them when we pickle the trained tagger.
        self._sentences = None

        self.model.average_weights()
        # Pickle as a binary file
        if save_loc is not None:
            with open(save_loc, 'wb') as fout:
                # changed protocol from -1 to 2 to make pickling Python 2 compatible
                pickle.dump((self.model.weights, self.tagdict, self.classes), fout, 2)

    def load(self, loc):
        '''
        :param loc: Load a pickled model at location.
        :type loc: str
        '''

        self.model.weights, self.tagdict, self.classes = load(loc)
        self.model.classes = self.classes

    def normalize(self, word):
        '''
        Normalization used in pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS

        :rtype: str
        '''
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        '''Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained.
        '''

        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i - 1])
        add('i-1 suffix', context[i - 1][-3:])
        add('i-2 word', context[i - 2])
        add('i+1 word', context[i + 1])
        add('i+1 suffix', context[i + 1][-3:])
        add('i+2 word', context[i + 2])
        return features

    def _make_tagdict(self, sentences):
        '''
        Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.
        '''
        counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            self._sentences.append(sentence)
            for word, tag in sentence:
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (mode / n) >= ambiguity_thresh:
                self.tagdict[word] = tag

class HMM(object):
    def __init__(self,corpus):
        self.corpus_tags_words = []
        for sent in corpus:
            self.corpus_tags_words.append(("START", "START"))
            self.corpus_tags_words.extend([(tag,word) for (word,tag) in sent])
            self.corpus_tags_words.append(("END", "END"))
        # conditional frequency distribution
        self.cfd_tagwords = nltk.ConditionalFreqDist(self.corpus_tags_words)
        # conditional probability distribution
        self.cpd_tagwords = nltk.ConditionalProbDist(self.cfd_tagwords, nltk.MLEProbDist)
        corpus_tags = [tag for (tag, word) in self.corpus_tags_words]
        self.distinct_tags = set(corpus_tags)
        # count(t{i-1} ti)
        # bigram的意思是 前后两个一组，联在一起
        self.cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(corpus_tags))
        self.cpd_tags = nltk.ConditionalProbDist(self.cfd_tags, nltk.MLEProbDist)

    def proc_tagwords(self, tag, word):
        """若词典里没有对应，则让其概率赋予很小的数"""
        if self.cpd_tagwords[tag].prob(word) == 0:
            return 1 / 1161192
        else:
            return self.cpd_tagwords[tag].prob(word)

    def tag(self,sentence):
        pre_viterbi = {}
        back_point = [];
        back_point.append({})
        ###START
        for tag in self.distinct_tags:
            if tag == "START":
                continue
            else:
                pre_viterbi[tag] = self.cpd_tags["START"].prob(tag) * self.proc_tagwords(tag, sentence[0])
                back_point[0][tag] = "START"
        ###COMMON
        for i in range(1, len(sentence)):
            back_point.append({})
            temp = {}
            for tag in self.distinct_tags:
                if tag == "START":
                    continue
                else:
                    best_pre = max(pre_viterbi.keys(),
                                   key=lambda tag_1: pre_viterbi[tag_1] * self.cpd_tags[tag_1].prob(tag) * self.proc_tagwords(tag, sentence[i]))
                    back_point[i][tag] = best_pre
                    temp[tag] = pre_viterbi[best_pre] * self.cpd_tags[best_pre].prob(tag) * self.proc_tagwords(tag,sentence[i])
            pre_viterbi = temp  # 更新pre_viterbi,使其是下一个指标的pre_viterbi
        ###END
        best_pre = max(pre_viterbi.keys(), key=lambda tag_1: pre_viterbi[tag_1] * self.cpd_tags[tag_1].prob("END"))
        back_point.append({})
        back_point[len(sentence)]["END"] = best_pre
        final_viterbi = {}
        final_viterbi["END"] = pre_viterbi[best_pre] * self.cpd_tags[best_pre].prob("END")

        back_point.reverse()
        sentence_tag = []
        index = "END"
        for i in range(len(back_point)):
            index = back_point[i][index]
            sentence_tag.append(index)
        sentence_tag.reverse()
        result=[]
        for i in range(len(sentence)):
            result.append([sentence[i],sentence_tag[i+1]])
        return result



