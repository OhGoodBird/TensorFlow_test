import os
import tensorflow as tf
from data_loader import DataLoader


'''
some code refer to https://github.com/guillaumegenthial/sequence_tagging
'''

class RNNModel(object):
    def __init__(self, config):
        self.config = config
        self.sess = None
        self.saver = None
    
    def add_placeholders(self):
        # shape = (batch size, max length of sentence)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                            name='word_ids')

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                    name='sequence_lengths')
        
        self.batch_size = tf.placeholder(tf.int32, shape=[None],
                                    name='batch_size')
                                    
        # shape = (batch size)
        self.labels = tf.placeholder(tf.int64, shape=[None],
                                    name='labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                    name='dropout_keep_prob')
                                    
                                    
    def add_word_embeddings_op(self):
        with tf.variable_scope('embedding_layer'):
            # shape = (vocab size, word embedding dim)
            #word_embedding_matrix = tf.Variable(
            #                            self.config.embedding_matrix,
            #                            name="embedding_matrix",
            #                            dtype=tf.float32,
            #                            trainable=True)
            word_embedding_matrix = tf.get_variable(
                                        name="embedding_matrix",
                                        dtype=tf.float32,
                                        shape=[len(self.config.word2id), self.config.hidden_size])
                                        
            word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix,
                                        self.word_ids, name="word_embeddings")
            
            word_embeddings =  tf.nn.dropout(word_embeddings, self.config.dropout_keep_prob)
            return word_embeddings
    
    
    #def add_fc_layer1(self, input):
    #    with tf.variable_scope('fc_layer1'):
    #        w = tf.Variable(tf.random_normal([self.config.wordemb_dim, self.config.hidden_size]), name='fc_layer1_w')
    #        b = tf.Variable(tf.random_normal([self.config.hidden_size]), name='fc_layer1_b')
    #        fc_layer1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(input, w), b)), self.config.dropout_keep_prob)
    #        return fc_layer1
            
    
    def add_rnn_layer(self, input):
        with tf.variable_scope('rnn_layer'):
            rnn_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size, name='rnn_cell')
            #rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size, name='rnn_cell')
            initial_state = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            output, state = tf.nn.dynamic_rnn(rnn_cell, input, sequence_length=self.sequence_lengths, initial_state=initial_state, dtype=tf.float32)
            #output, state = tf.nn.dynamic_rnn(rnn_cell, input, sequence_length=self.sequence_lengths, dtype=tf.float32)
            #output = tf.transpose(output, [1, 0, 2])[-1]
            #self.we = tf.gather(tf.transpose(output, [1, 0, 2]), self.sequence_lengths)
            output = tf.transpose(output, [1, 0, 2])
            #self.we = tf.gather(output, self.sequence_lengths-1)
            output = tf.gather_nd(output, tf.reshape(self.sequence_lengths-1, [-1, 1]))[-1]
            self.we = output
            return output
    
    
    def add_final_layer(self, input):
        with tf.variable_scope('final_layer'):
            w = tf.Variable(tf.random_normal([self.config.hidden_size, self.config.label_dim]), name='final_layer_w')
            b = tf.Variable(tf.random_normal([self.config.label_dim]), name='final_layer_b')
            logits = tf.matmul(input, w) + b
            return logits
            
    
    def add_train_op(self, input):
        self.probs = tf.nn.softmax(input)
        self.ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=input, labels=self.labels)
        self.loss_sum = tf.reduce_sum(self.ce)
        self.loss_avg = tf.reduce_mean(self.ce)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config.learning_rate)
        if(self.config.clip > 0):
            grads, vs = zip(*optimizer.compute_gradients(self.loss_avg))
            grads, gnorm  = tf.clip_by_global_norm(grads, self.config.clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(self.loss_avg)
        
        self.pred_label = tf.argmax(self.probs, 1)
        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.labels)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
    
    def initialize_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        
    def restore_session(self, dir_model):
        self.saver.restore(self.sess, dir_model)


    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)
        
        
    def build(self):
        self.add_placeholders()
        word_embedding = self.add_word_embeddings_op()
        #fc1 = self.add_fc_layer1(word_embedding)
        rnn_output = self.add_rnn_layer(word_embedding)
        logits = self.add_final_layer(rnn_output)
        self.add_train_op(logits)
        
        self.initialize_session()
        
    
    '''
    reference: https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
    '''
    def get_num_params(self):
        total_parameters = 0
        #iterating over all variables
        for variable in tf.trainable_variables():  
            local_parameters=1
            shape = variable.get_shape()  #getting shape of a variable
            for i in shape:
                local_parameters*=i.value  #mutiplying dimension values
            total_parameters+=local_parameters
        return total_parameters

        
    def train(self):
        import numpy as np
        dl = DataLoader(self.config.train_path, self.config.word2id, self.config.label2id, self.config.max_len)
        for epoch in range(self.config.epoch_num):
            epoch += 1
            epoch_loss = 0
            for batch_data in dl.getBatchData(batch_size=self.config.batch_size):
                iter = batch_data[0]
                total_iter = batch_data[1]
                y = batch_data[2]
                x = batch_data[3]
                lengths = batch_data[4]
                batch_size = np.array([len(y)])
                
                fd = {self.word_ids: x,
                      self.sequence_lengths: lengths,
                      self.labels: y,
                      self.batch_size: batch_size,
                      self.dropout_keep_prob: self.config.dropout_keep_prob}
                _, loss, accuracy = self.sess.run([self.train_op, self.loss_sum, self.accuracy], feed_dict=fd)
                epoch_loss += loss
                #print('epoch={epoch}\t{iter}/{total_iter}\tloss={loss}\taccuracy={accuracy}'.format(epoch=epoch, \
                #                                                                                     iter=iter, \
                #                                                                                     total_iter=total_iter, \
                #                                                                                     loss=loss, \
                #                                                                                     accuracy=accuracy), flush=True)
            print('epoch {}/{}, total loss = {}'.format(epoch, self.config.epoch_num, epoch_loss))
            self.test()
            print('')
        
    
    def test(self):
        import numpy as np
        dl = DataLoader(self.config.train_path, self.config.word2id, self.config.label2id, self.config.max_len)
        total_correct = 0
        total_num = dl.data_num
        fp = open('t.txt', 'w', encoding='utf8')
        for batch_data in dl.getBatchData(batch_size=2, shuffle=False):
            iter = batch_data[0]
            total_iter = batch_data[1]
            y = batch_data[2]
            x = batch_data[3]
            lengths = batch_data[4]
            batch_size = np.array([len(y)])
            
            fd = {self.word_ids: x,
                  self.sequence_lengths: lengths,
                  self.labels: y,
                  self.batch_size: batch_size,
                  self.dropout_keep_prob: 1.0}
            correct_num, probs, pred_label = self.sess.run([self.correct_num, self.probs, self.pred_label], feed_dict=fd)
            total_correct += correct_num
            for i in range(len(y)):
                s = ' '.join([self.config.id2word[_] for _ in filter(lambda a: a != 0, x[i].tolist())])
                fp.write('{}, {}, {}, {}\n'.format(y[i], pred_label[i], probs[i], s))
            fp.write('\n')
                
            #[we] = self.sess.run([self.we], feed_dict=fd)
            #fp.write('{}\n'.format(we.shape))
            #fp.write('{}\n'.format(we))
        fp.close()
        accuracy = total_correct *100 / total_num
        print('accuracy = %.2f %% (%d / %d)' %(accuracy, total_correct, total_num))

