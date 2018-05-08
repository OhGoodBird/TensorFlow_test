from utils import getWordEmbedding, getLabelMap

class Config(object):
    def __init__(self):
        
        self.word2id, self.id2word, self.embedding_matrix = getWordEmbedding(self.w2v_path, num_words=-1)
        self.wordemb_dim = self.embedding_matrix.shape[1]
        
        self.label2id, self.id2label = getLabelMap(self.label_path)
        self.label_dim = len(self.label2id)
        
        
    w2v_path = 'word2vec_model/blogwiki_size200_alpha01_iter20.model'

    label_path = 'data/label_list.txt'
    
    train_path = 'data/train_wb_utf8.csv'
    valid_path = 'data/valid_wb_utf8.csv'
    test_path = 'data/test_wb_utf8.csv'
    
    dir_model = 'model'
    
    max_len = 32
    hidden_size = 50
    epoch_num = 1000
    batch_size = 4
    dropout_keep_prob = 1.0
    learning_rate = 0.001
    clip = -1
    