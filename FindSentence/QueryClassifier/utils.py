#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def getWordEmbedding(word2vec_model_path, extra_word=[], num_words=-1):
    import numpy as np
    from gensim.models import Word2Vec
    from collections import OrderedDict
    w2v_mdl = Word2Vec.load(word2vec_model_path)    
    word_list = []
    if(len(w2v_mdl.wv.vocab) > num_words):
        vocabs = OrderedDict()
        for word in w2v_mdl.wv.vocab:
            vocabs[word] = w2v_mdl.wv.vocab[word].count
        vocabs = OrderedDict(sorted(vocabs.items(), key=lambda x: x[1], reverse=True))
        fp_out = open('debug.txt', 'w', encoding='utf8')
        for word in vocabs:
            fp_out.write('%s\t%d\n' %(word, vocabs[word]))
        fp_out.close()
        cnt = 0
        for word in vocabs:
            cnt += 1
            word_list.append(word)
            if(num_words != -1 and cnt >= num_words):
                break
    else:
        for word in model.wv.vocab:
            word_list.append(word)
    
    num_word = len(word_list) + len(extra_word)
    embedding_dim = w2v_mdl.layer1_size
        
    word2id = {}
    id2word = {}
    embedding_matrix = np.zeros([num_word, embedding_dim])
    for i, word in enumerate(extra_word):
        word2id[word] = i
        id2word[i] = word
    for i, word in enumerate(word_list):
        word2id[word] = i + len(extra_word)
        id2word[i+len(extra_word)] = word
        embedding_matrix[i+len(extra_word)] = w2v_mdl.wv[word]
    return word2id, id2word, embedding_matrix
    
