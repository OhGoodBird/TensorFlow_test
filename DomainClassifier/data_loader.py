
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd

class DataLoader():
    def __init__(self, filepath, word2id, label2id, max_len=32):
        self.index = 0
        self.data_num = 0
        self.word2id = word2id
        self.label2id = label2id
        self.data = pd.DataFrame()
        
        # load data from file
        dataframe = pd.read_csv(filepath)
        
        self.data_num = len(dataframe)
        
        # convert label to label id
        y = [label2id[x] for x in dataframe['domain_intent']]
        
        # convert word to word id in sentence
        def _sentWord2Id(s, word2id):
            return [word2id[x] for x in s.split(' ') if x in word2id]
        x = [_sentWord2Id(w, word2id) for w in dataframe['sentence']]
        
        # get sentence len
        length = [len(s) for s in x]
        
        # pad sentence to max_len
        def _sentPadding(s, max_len, word2id):
            if(len(s) >= max_len):
                return s[:max_len]
            else:
                pad_num = max_len - len(s)
                for i in range(pad_num):
                    s.append(word2id['_PAD_'])
                return s
        x = [_sentPadding(w, max_len, word2id) for w in x]
        #x = [' '.join([str(_) for _ in s]) for s in x]
        #print(x)
                
        self.data['domain_intent'] = dataframe['domain_intent']
        self.data['y'] = np.array(y)
        self.data['sentence'] = dataframe['sentence']
        self.data['x'] = x
        self.data['length'] = np.array(length)
        
        
        
    '''
    Return : current iteration, total iteration, label(ont-hot), sample
    '''
    def getBatchData(self, batch_size=16, shuffle=True):
        if(batch_size > self.data_num):
            raise ValueError('batch_size (%d) > data_num (%d)' %(batch_size, self.data_num))
        
        if(shuffle):
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            
        batch_num = int(math.ceil(self.data_num / batch_size))
        for i in range(batch_num):
            start = i * batch_size
            end = start + batch_size
            if(end > self.data_num):
                end = self.data_num
                
            # dtype object => dtype np.float32
            x = np.array([[int(_) for _ in s] for s in self.data['x'].values[start:end]], dtype=np.int32)
            
            yield i+1,                   batch_num,                   self.data['y'].values[start:end],                   x,                   self.data['length'].values[start:end]


# In[2]:


def main():
    from utils import getWordEmbedding, getLabelMap
    
    w2v_path = 'word2vec_model/blogwiki_size200_alpha01_iter20.model'
    (word2id, id2word, embedding_matrix) = getWordEmbedding(w2v_path)
    wordemb_dim = embedding_matrix.shape[1]
    
    label_path = 'data/label_list.txt'
    label2id, id2label = getLabelMap(label_path)
    
    dl = DataLoader('data/test_wb_utf8.csv', word2id, label2id, max_len=16)
    
    for i, n, y, x, l in dl.getBatchData(batch_size=16, shuffle=False):
        print('iter = %d'%(i))
        for i in range(len(y)):
            print('\t{}, {}, {}'.format(y[i], x[i], l[i]))
            #break
    #print('===================================')
    #for i, n, y, x, l in dl.getBatchData(batch_size=16):
    #    print('iter = %d'%(i))
    #    for i in range(len(y)):
    #        print('\t{}, {}, {}'.format(y[i], x[i], l[i]))
    #        #break
            
if(__name__ == '__main__'):
    main()

