import collections
import re
from copy import copy
from diff2.corpus_parser import CorpusParser


class POSTagger(object):
    def __init__(self, data_io=(), eager=False):
        self.corpus_parser = CorpusParser()
        self.data_io = data_io
        self.trained = False
        if eager:
            self.train()
            self.trained = True

    def train(self):
        if not self.trained:
            self.tags = set()
            self.tag_combos = collections.defaultdict(int)
            self.tag_frequencies = collections.defaultdict(int)
            self.word_tag_combos = collections.defaultdict(int)
            for io in self.data_io:
                for line in io:
                    for ngram in self.corpus_parser.parse(line):
                        self.write(ngram)
            self.trained = True

    def write(self, ngram):
        if ngram[0].tag == 'START':
            self.tag_frequencies["START"] += 1
            self.word_tag_combos['START/START'] += 1
        self.tags.add(ngram[-1].tag)
        self.tag_frequencies[ngram[-1].tag] += 1
        combo = ngram[0].tag + '/' + ngram[-1].tag
        self.word_tag_combos[combo] += 1
        combo = ngram[0].tag + '/' + ngram[-1].tag
        self.tag_combos[combo] += 1

    def tag_propability(self, previous_tag, current_tag):
        # count(previous_tag, current_tag)/ count(previous_tag)
        denom = self.tag_frequencies[previous_tag]
        if denom == 0:
            return 0
        else:
            return self.tag_combos[previous_tag + '/' + current_tag] / float(denom)

    def word_tag_propability(self, word, tag):
        denom = self.tag_frequencies[tag]
        if denom == 0:
            return 0
        else:
            return self.word_tag_combostag_combos[word + '/' + tag] / float(denom)

    def propability_of_word_tag(self, words, tags):
        if len(words) != len(tags):
            raise ValueError('Words and tags have to be the same length')
        length = len(words)
        propability = 1.0
        for i in range(1, length):
            propability += self.tag_propability(tags[i - 1], tags[i]) * self.word_tag_propability(words[i], tags[i])
        return propability

    def viterbi(self, sentence):
        sentence = re.sub(r'([\.\?!])', r' \1', sentence)
        parts = re.split(r'\s+', sentence)
        last_viterbi = {}
        backpointers = ['START']
        for tag in self.tag:
            if tag == 'START':
                continue
            else:
                propabilty = self.tag_propability('START', tag) * self.word_tag_propability(parts[0], tag)
                if propabilty > 0:
                    last_viterbi[tag] = propabilty
            if len(last_viterbi) > 0:
                backpointer = max(last_viterbi, key=(lambda key: last_viterbi[key]))
            else:
                backpointer = max(self.tag_frequencies, key=(lambda key: self.tag_frequencies[key]))
            backpointers.append(backpointer)
        for part in parts[1:]:
            viterbi = {}
            if tag == 'START':
                continue
            if len(last_viterbi) == 0:
                break
            best_tag = max(last_viterbi, key=(lambda prev_tag: last_viterbi[prev_tag] *
                                                               self.tag_propability(prev_tag, tag) *
                                                               self.word_tag_propability(part, tag)))
            propabilty = last_viterbi[best_tag] * \
                         self.tag_propability(best_tag, tag) * self.word_tag_propability(part, tag)
            if propabilty > 0:
                viterbi[tag] = propabilty
            last_viterbi = viterbi
            if len(last_viterbi) > 0:
                backpointer = max(last_viterbi, key=(lambda key: last_viterbi[key]))
            else:
                backpointer = max(self.tag_frequencies, key=(lambda key: self.tag_frequencies[key]))
            backpointers.append(backpointer)
        return backpointers
