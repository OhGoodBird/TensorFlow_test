
# coding: utf-8

# In[1]:


#!/usr/bin python3

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec


# In[2]:


# load word2vec model
w2v_model_path = 'word2vec_model/word2vec.model'

w2v_mdl = Word2Vec.load(w2v_model_path)
num_word = len(w2v_mdl.wv.vocab)
embedding_dim = w2v_mdl.layer1_size
print('number of vocab = %d' %(num_word))
print('embedding_dim = %d' %(embedding_dim))


# In[3]:


# get embedding_matrix, word2id and id2word
word2id = {}
id2word = {}
embedding_matrix = np.zeros([num_word, embedding_dim])
for i, word in enumerate(w2v_mdl.wv.vocab):
    word2id[word] = i
    id2word[i] = word
    embedding_matrix[i] = w2v_mdl.wv[word]


# In[4]:


# sentences that we would like to get word embedding vector
# all sentence must have same length (i.e. do padding first)
sentences = [['i', 'love', 'you'], ['this', 'is', 'book']]
sentences_id = np.array([[word2id[word] for word in _] for _ in sentences])
print('sentences_id =\n{}'.format(sentences_id))
print('sentences_id.shape = {}'.format(sentences_id.shape))


# In[5]:


init_embedding_W = tf.constant_initializer(embedding_matrix)
word_embeddings = tf.get_variable('word_embeddings',
                                  shape=[embedding_matrix.shape[0], embedding_matrix.shape[1]], 
                                  initializer=init_embedding_W)


# In[6]:


sentences_embedding = tf.placeholder(dtype=tf.int32, shape=(None, None), name='sentences_embedding')
sentences_vec = tf.nn.embedding_lookup(word_embeddings, sentences_embedding, name='sentences_vec')


# In[7]:


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run(sentences_vec, feed_dict={sentences_embedding: sentences_id}))

