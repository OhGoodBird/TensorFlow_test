#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def getWordEmbedding(word2vec_model_path, extra_word=['_PAD_'], num_words=-1):
    import numpy as np
    from gensim.models import Word2Vec
    from collections import OrderedDict
    # load w2v model
    w2v_mdl = Word2Vec.load(word2vec_model_path)    
    
    # get words(vocabulary) in model
    word_list = []
    vocabs = OrderedDict()
    for word in w2v_mdl.wv.vocab:
        vocabs[word] = w2v_mdl.wv.vocab[word].count
    vocabs = OrderedDict(sorted(vocabs.items(), key=lambda x: x[1], reverse=True))
    cnt = 0
    for word in vocabs:
        cnt += 1
        word_list.append(word)
        if(num_words != -1 and cnt >= num_words):
            break
    
    num_word = len(word_list) + len(extra_word)
    embedding_dim = w2v_mdl.layer1_size
        
    word2id = {}
    id2word = {}
    embedding_matrix = np.zeros([num_word, embedding_dim])
    # extra_word part
    for i, word in enumerate(extra_word):
        word2id[word] = i
        id2word[i] = word
    # model words part
    for i, word in enumerate(word_list):
        word2id[word] = i + len(extra_word)
        id2word[i+len(extra_word)] = word
        embedding_matrix[i+len(extra_word)] = w2v_mdl.wv[word]
    return word2id, id2word, embedding_matrix
    
    
def getLex(lex_path, extra_word=['_PAD_']):
    word_list = []
    fp = open(lex_path, 'r', encoding='utf8')
    for line in fp:
        line = line.strip()
        word_list.append(line)
    fp.close()
    word2id = {}
    id2word = {}
    # extra_word part
    for i, word in enumerate(extra_word):
        word2id[word] = i
        id2word[i] = word
    # model words part
    for i, word in enumerate(word_list):
        word2id[word] = i + len(extra_word)
        id2word[i+len(extra_word)] = word
    return word2id, id2word
    
    
def getLabelMap(filepath):
    labels = []
    label2id = {}
    id2label = {}
    with open(filepath, 'r', encoding='utf8') as fp:
        for line in fp:
            line = line.strip()
            if line not in labels:
                labels.append(line)
    for x in range(len(labels)):
        label2id[labels[x]] = x
        id2label[x] = labels[x]
        
    return (label2id, id2label)

    
def chkDirExistOrCreate(folder_path):
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.chmod(folder_path, 0o777)
                    
                    