from utils import getWordEmbedding, getLex, getLabelMap, chkDirExistOrCreate

class Config(object):
    def __init__(self):
        if(self.use_pretrained_word_embedding):
            self.word2id, self.id2word, self.embedding_matrix = getWordEmbedding(self.w2v_path, extra_word=['_PAD_'], num_words=-1)
            self.wordemb_dim = self.embedding_matrix.shape[1]
        else:
            self.word2id, self.id2word = getLex(self.lex_path, extra_word=['_PAD_'])
        
        self.label2id, self.id2label = getLabelMap(self.label_path)
        self.label_dim = len(self.label2id)
        
        
    w2v_path = 'word2vec_model/blogwiki_size200_alpha01_iter20.model'
    use_pretrained_word_embedding = False
    if(use_pretrained_word_embedding):
        w2v_path = 'word2vec_model/blogwiki_size200_alpha01_iter20.model'
        replaceDigit = '#'
    else:
        lex_path = 'data/input/lex.txt'
        replaceDigit = '#'
        wordemb_dim = 50

    label_path = 'data/output/label_list.txt'
    
    train_path = 'data/output/data_train.csv'
    valid_path = 'data/output/data_valid.csv'
    test_path = 'data/output/data_test.csv'
    
    save_model_path = 'model'
    output_result_path = 'result'
    chkDirExistOrCreate(save_model_path)
    chkDirExistOrCreate(output_result_path)
    
    
    optimizer = 'sgd'
    
    max_len = 32
    hidden_size = 30
    epoch_num = 1000
    batch_size = 32
    dropout_keep_prob = 0.8
    learning_rate = 0.1
    clip = 5.0
    lr_decay_rate = 0.8
    lr_decay_epoch = 3
    early_stop_epoch = 10
    
    