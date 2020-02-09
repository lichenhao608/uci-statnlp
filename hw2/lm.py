#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)


class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences. """
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        vocab_set = set(self.vocab())
        words_set = set([w for s in corpus for w in s])
        numOOV = len(words_set - vocab_set)
        return pow(2.0, self.entropy(corpus, numOOV))

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1  # for EOS
            sum_logprob += self.logprob_sentence(s, numOOV)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence, numOOV):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i], numOOV)
        p += self.cond_logprob('END_OF_SENTENCE', sentence, numOOV)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass


class Unigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')
        self.inc_word('START_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
            return self.model[word]
        else:
            return self.lunk_prob-log(numOOV, 2)

    def vocab(self):
        return self.model.keys()


class Bigram(LangModel):
    def __init__(self, unk_prob=1e-4, gamma=0, smooth=1):
        self.model = {}
        self.lam = smooth
        self.num_gamma = gamma
        self.filter = Unigram()

        self.vocab_set = set()
        self.rare = set()
        self.num_count = {}
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, previous, w):
        if previous not in self.model:
            self.model[previous] = {}
        if w not in self.model[previous]:
            self.model[previous][w] = 0.0
        self.model[previous][w] += 1.0

    def fit_sentence(self, sentencce):
        for i, w in enumerate(sentencce):
            if i == 0:
                self.inc_word('START_OF_SENTENCE', w)
            else:
                self.inc_word(sentencce[i - 1], w)
            self.filter.inc_word(w)
        if len(sentencce) >= 1:
            self.inc_word(sentencce[-1], 'END_OF_SENTENCE')
        elif len(sentencce) == 0:
            self.inc_word('START_OF_SENTENCE', 'END_OF_SENTENCE')
        self.filter.inc_word('END_OF_SENTENCE')

    def norm(self):

        for key in self.filter.model:
            if self.filter.model[key] >= self.num_gamma:
                self.vocab_set.add(key)
            else:
                self.rare.add(key)
        self.vocab_set.add('START_OF_SENTENCE')
        self.vocab_set.add('UNK')

        if self.rare:
            self.filter.model['UNK'] = 0.0
            for w in self.rare:
                self.filter.model['UNK'] += self.filter.model[w]
                del self.filter.model[w]

        self.filter.norm()

        # handling rare conditional words
        for i, previous in enumerate(list(self.model.keys())):
            if previous in self.rare:
                prev = 'UNK'
                if prev not in self.model:
                    self.model[prev] = {}

                for w, num in self.model[previous].items():
                    if w not in self.model[prev]:
                        self.model[prev][w] = 0.0
                    self.model[prev][w] += num

                del self.model[previous]

        # make sure that when testing there is situation that is unk,unk
        if 'UNK'not in self.model:
            self.model['UNK'] = {}

        # smoothing
        for i, previous in enumerate(list(self.model.keys())):
            tot = sum(self.model[previous].values())

            # unknown
            for w in self.rare:
                if w in self.model[previous]:
                    if 'UNK' not in self.model[previous]:
                        self.model[previous]['UNK'] = 0.0
                    self.model[previous]['UNK'] += self.model[previous][w]
                    del self.model[previous][w]

            # make sure that when testing there is situation that is unk
            if 'UNK' not in self.model[previous]:
                self.model[previous]['UNK'] = 0.0

            self.num_count[previous] = tot

    def cond_logprob(self, word, previous, numOOV):
        # print(numOOV)

        if len(previous) == 0:
            prev = 'START_OF_SENTENCE'
        else:
            prev = previous[-1]
            if prev not in self.vocab_set:
                prev = 'UNK'

        # apply smoothing here
        if self.lam:
            if prev in self.model:
                if word in self.model[prev]:
                    return log(self.model[prev][word] + self.lam, 2) - \
                        log(self.lam * (len(self.vocab_set) - 1) +
                            self.num_count[prev], 2)
                elif word in self.vocab_set:
                    return log(self.lam, 2) - log(self.lam * (len(self.vocab_set) - 1) + self.num_count[prev], 2)
                else:
                    return log(self.model[prev]['UNK'] + self.lam, 2) - \
                        log(self.lam * (len(self.vocab_set) - 1) + self.num_count[prev], 2) - \
                        log(numOOV, 2)
            else:
                return self.filter.cond_logprob(word, previous, numOOV)
        # only for no smoothing
        else:
            if word in self.model[prev]:
                return log(self.model[prev][word], 2) - log(self.num_count[prev], 2)
            else:
                try:
                    return self.lunk_prob - log(numOOV, 2)
                except ValueError:
                    print(numOOV)
                    print(prev)
                    print(previous[len(previous) - 2:])
                    print(word)

    def vocab(self):
        return list(self.vocab_set - set(['START_OF_SENTENCE']))


class Trigram(LangModel):
    def __init__(self, unk_prob=1e-4, gamma=0, smooth=1, lambda1=0.65, lambda2=0.25):
        self.model = {}
        self.lam = smooth
        self.num_gamma = gamma
        self.w1 = lambda1
        self.w2 = lambda2
        self.w3 = 1 - self.w1 - self.w2

        self.w11 = self.w1 / (1 - self.w3)
        self.w22 = self.w2 / (1 - self.w3)

        self.num_count = {}

        self.bi = Bigram(gamma=gamma, smooth=smooth)

    def inc_word(self, previous, w):
        if previous not in self.model:
            self.model[previous] = {}
        if w not in self.model[previous]:
            self.model[previous][w] = 0.0
        self.model[previous][w] += 1.0

    def fit_sentence(self, sentencce):
        for i, w in enumerate(sentencce):
            if i == 0:
                self.inc_word(('START_OF_SENTENCE', 'START_OF_SENTENCE'), w)
            elif i == 1:
                self.inc_word(('START_OF_SENTENCE', sentencce[i - 1]), w)
            else:
                self.inc_word((sentencce[i - 2], sentencce[i - 1]), w)
        if len(sentencce) >= 2:
            self.inc_word((sentencce[-2], sentencce[-1]), 'END_OF_SENTENCE')
        elif len(sentencce) == 0:
            self.inc_word(
                ('START_OF_SENTENCE', 'START_OF_SENTENCE'), 'END_OF_SENTENCE')
        else:
            self.inc_word(
                ('START_OF_SENTENCE', sentencce[0]), 'END_OF_SENTENCE')
        self.bi.fit_sentence(sentencce)
        # print(self.model)

    def norm(self):
        # import time
        self.bi.norm()

        # handling rare conditional words
        # t = time.time()
        for i, previous in enumerate(list(self.model.keys())):
            if previous[0] in self.bi.rare or previous[1] in self.bi.rare:
                prev = tuple(
                    w if w in self.bi.vocab_set else 'UNK' for w in previous)
                if prev not in self.model:
                    self.model[prev] = {}

                for w, num in self.model[previous].items():
                    if w not in self.model[prev]:
                        self.model[prev][w] = 0.0
                    self.model[prev][w] += num

                del self.model[previous]

        # smoothing
        for i, previous in enumerate(list(self.model.keys())):
            # unknown
            for w in self.bi.rare:
                if w in self.model[previous]:
                    if 'UNK' not in self.model[previous]:
                        self.model[previous]['UNK'] = 0.0
                    self.model[previous]['UNK'] += self.model[previous][w]
                    del self.model[previous][w]
                    assert w not in self.model[previous]

            # make sure that when testing there is situation that is unk
            if 'UNK' not in self.model[previous]:
                self.model[previous]['UNK'] = 0.0

            self.num_count[previous] = sum(self.model[previous].values())

    def _cond_logprob(self, word, previous, numOOV):
        if len(previous) < 2:
            for _ in range(2 - len(previous)):
                previous = ['START_OF_SENTENCE'] + previous

        # prev = tuple(previous[len(previous)-2:])
        prev = tuple(
            w if w in self.bi.vocab_set else 'UNK' for w in previous[-2:])
        assert prev[0] in self.bi.vocab_set and prev[1] in self.bi.vocab_set

        # apply smoothing here
        if self.lam:
            if prev in self.model:
                if word in self.model[prev]:
                    return log(self.model[prev][word] + self.lam, 2) - \
                        log(self.lam * (len(self.bi.vocab_set) - 1) +
                            self.num_count[prev], 2)
                elif word in self.bi.vocab_set:
                    return log(self.lam, 2) - \
                        log(self.lam * (len(self.bi.vocab_set) - 1) +
                            self.num_count[prev], 2)
                else:
                    return log(self.model[prev]['UNK'] + self.lam, 2) - \
                        log(self.lam * (len(self.bi.vocab_set) - 1) + self.num_count[prev], 2) - \
                        log(numOOV, 2)
            else:
                return self.bi.cond_logprob(word, prev, numOOV)
        # only for no smoothing
        else:
            if word in self.model[prev]:
                return log(self.model[prev][word], 2) - log(self.num_count[prev], 2)
            else:
                try:
                    return self.lunk_prob - log(numOOV, 2)
                except ValueError:
                    print(numOOV)
                    print(prev)
                    print(previous[len(previous) - 2:])
                    print(word)

    def cond_logprob(self, word, previous, numOOV):
        if not numOOV and word == 'UNK':
            return self.w11 * self._cond_logprob(word, previous, numOOV) + \
                self.w22 * self.bi.cond_logprob(word, previous, numOOV)
        else:
            return self.w1 * self._cond_logprob(word, previous, numOOV) + \
                self.w2 * self.bi.cond_logprob(word, previous, numOOV) + \
                self.w3 * \
                self.bi.filter.cond_logprob(word, previous, numOOV)

    def vocab(self):
        return self.bi.vocab()
