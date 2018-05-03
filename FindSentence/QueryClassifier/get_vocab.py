#!/usr/bin/env python3

import argparse
from gensim.models import Word2Vec
from collections import OrderedDict
from operator import itemgetter 

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('vocab_out', type=str)
args = parser.parse_args()

model = Word2Vec.load(args.model)

vocabs = OrderedDict()
for word in model.wv.vocab:
    vocabs[word] = model.wv.vocab[word].count

fp_out = open(args.vocab_out, 'w')
vocabs = OrderedDict(sorted(vocabs.items(), key=lambda x: x[1], reverse=True))
for word in vocabs:
    fp_out.write('%s\t%d\n' %(word, vocabs[word]))

fp_out.close()