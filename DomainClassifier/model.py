#!/usr/bin/env python3
# coding=UTF-8

import os
import time
import numpy as np
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
        
        self.batch_size = tf.placeholder(tf.int32, shape=[],
                                    name='batch_size')
                                    
        # shape = (batch size)
        self.labels = tf.placeholder(tf.int64, shape=[None],
                                    name='labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                    name='dropout_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32,
                                    name='learning_rate')
                                    
                                    
    def add_word_embeddings_op(self):
        with tf.variable_scope('embedding_layer'):
            # shape = (vocab size, word embedding dim)
            if(self.config.use_pretrained_word_embedding):
                word_embedding_matrix = tf.Variable(
                                            self.config.embedding_matrix,
                                            name="embedding_matrix",
                                            dtype=tf.float32,
                                            trainable=False)
            else:
                word_embedding_matrix = tf.get_variable(
                                            name="embedding_matrix",
                                            dtype=tf.float32,
                                            shape=[len(self.config.word2id), self.config.wordemb_dim])
                                        
            word_embeddings = tf.nn.embedding_lookup(word_embedding_matrix,
                                        self.word_ids, name="word_embeddings")
            
            word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_keep_prob)
            return word_embeddings
    
    
    def add_fc_layer1(self, input):
        with tf.variable_scope('fc_layer1'):
            input = tf.reshape(input, [-1, self.config.wordemb_dim])
            w = tf.Variable(tf.random_normal([self.config.wordemb_dim, self.config.hidden_size]), name='fc_layer1_w')
            b = tf.Variable(tf.random_normal([self.config.hidden_size]), name='fc_layer1_b')
            fc_layer1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input, w) + b), self.dropout_keep_prob)
            fc_layer1 = tf.reshape(fc_layer1, [-1, self.config.max_len, self.config.hidden_size])
            return fc_layer1
            
    
    def add_rnn_layer(self, input):
        with tf.variable_scope('rnn_layer'):
            rnn_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size, name='rnn_cell')
            #rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size, name='rnn_cell')
            initial_state = rnn_cell.zero_state(tf.convert_to_tensor(self.batch_size), dtype=tf.float32)
            output, state = tf.nn.dynamic_rnn(rnn_cell, input, sequence_length=self.sequence_lengths, initial_state=initial_state, dtype=tf.float32)
            #output = tf.transpose(output, [1, 0, 2])
            indices = tf.stack([tf.range(self.batch_size), self.sequence_lengths-1], axis=1)
            output = tf.gather_nd(output, indices)
            #self.we = output
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
        
        _optimizer = self.config.optimizer.lower() # lower to make sure
        if(_optimizer == 'adam'): # sgd method
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        elif(_optimizer == 'adagrad'):
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate)
        elif(_optimizer == 'adadelta'):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config.learning_rate)
        elif(_optimizer == 'sgd'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
        elif(_optimizer == 'rmsprop'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
        else:
            raise NotImplementedError("Unknown method {}".format(_optimizer))
        
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
        
        
    def restore_session(self, path):
        self.saver.restore(self.sess, path)
        
        
    def close_session(self):
        tf.reset_default_graph()
        self.sess.close()


    def save_session(self, path):
        self.saver.save(self.sess, path)
        
        
    def build(self):
        self.add_placeholders()
        word_embedding = self.add_word_embeddings_op()
        fc1 = self.add_fc_layer1(word_embedding)
        rnn_output = self.add_rnn_layer(fc1)
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

        
    def train(self, train_dl, valid_dl, test_dl):
        best_epoch = -1
        best_train_loss = float('inf')
        best_train_acc = -1.0
        best_valid_loss = float('inf')
        best_valid_acc = -1.0
        pre_loss = float('inf')
        loss_not_degrade_epoches = 0
        lr_decay = 1.0
        for epoch in range(self.config.epoch_num):
            time_start = time.time()
            epoch += 1
            epoch_loss = 0
            lr = self.config.learning_rate * lr_decay
            for batch_data in train_dl.getBatchData(batch_size=self.config.batch_size):
                iter = batch_data[0]
                total_iter = batch_data[1]
                y = batch_data[2]
                x = batch_data[3]
                lengths = batch_data[4]
                batch_size = len(y)
                
                fd = {self.word_ids: x,
                      self.sequence_lengths: lengths,
                      self.labels: y,
                      self.batch_size: batch_size,
                      self.dropout_keep_prob: self.config.dropout_keep_prob,
                      self.learning_rate: lr}
                _, accuracy = self.sess.run([self.train_op, self.accuracy], feed_dict=fd)
                
                fd = {self.word_ids: x,
                      self.sequence_lengths: lengths,
                      self.labels: y,
                      self.batch_size: batch_size,
                      self.dropout_keep_prob: 1.0}
                epoch_loss += self.sess.run(self.loss_sum, feed_dict=fd)
            print('Epoch {}/{}, lr = {}, total loss = {}'.format(epoch, self.config.epoch_num, lr, epoch_loss))
            train_acc, train_correct, train_num, train_loss = self.test(train_dl, 
                                                                        batch_size=1024, 
                                                                        output_file=os.path.join(self.config.output_result_path, 
                                                                            'epoch_%d_train.txt'%(epoch)))
            valid_acc, valid_correct, valid_num, valid_loss = self.test(valid_dl, 
                                                                        batch_size=1024, 
                                                                        output_file=os.path.join(self.config.output_result_path, 
                                                                            'epoch_%d_valid.txt'%(epoch)))
            test_acc, test_correct, test_num, test_loss = self.test(test_dl, 
                                                                    batch_size=1024, 
                                                                    output_file=os.path.join(self.config.output_result_path, 
                                                                        'epoch_%d_test.txt'%(epoch)))
            time_end = time.time()
            print('\tTrain accuracy = %.2f %% (%d / %d), loss = %.10f' %(train_acc, train_correct, train_num, train_loss))
            print('\tValid accuracy = %.2f %% (%d / %d), loss = %.10f' %(valid_acc, valid_correct, valid_num, valid_loss))
            print('\tTest  accuracy = %.2f %% (%d / %d), loss = %.10f' %(test_acc, test_correct, test_num, test_loss))
            print('\telapsed time = %.3f (sec.)' %(time_end - time_start))
            
            if(valid_loss < best_valid_loss):
                best_epoch = epoch
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc
                save_path = os.path.join(self.config.save_model_path, 'model.ckpt')
                self.save_session(save_path)
                print('\tModel saved in path: %s' %(save_path))
                
            if(epoch_loss < pre_loss):
                loss_not_degrade_epoches = 0
                pre_loss = epoch_loss
            else:
                loss_not_degrade_epoches += 1
                if((loss_not_degrade_epoches % self.config.lr_decay_epoch) == 0):
                    lr_decay *= self.config.lr_decay_rate
                if(loss_not_degrade_epoches >= self.config.early_stop_epoch):
                    print('loss doesn\'t decrease for %d epoches, stop training' %(loss_not_degrade_epoches))
                    break
                
            #print('loss_not_degrade_epoches = %d'%(loss_not_degrade_epoches))
            print('')
        
        print('Best Epoch = %d' %(best_epoch))
        print('Training Accuracy = %.2f %%, Loss = %.10f' %(best_train_acc, best_train_loss))
        print('Validation Accuracy = %.2f %%, Loss = %.10f' %(best_valid_acc, best_valid_loss))
        
    
    def test(self, dl, batch_size=1024, output_file=None):
        total_correct = 0
        total_num = dl.data_num
        
        if(output_file != None):
            fp = open(output_file, 'w', encoding='utf8')
            fp.write('{}, {}, {}, {}\n'.format('target', 'predict', 'probs', 'sentences'))
            
        for batch_data in dl.getBatchData(batch_size=batch_size, shuffle=False):
            iter = batch_data[0]
            total_iter = batch_data[1]
            y = batch_data[2]
            x = batch_data[3]
            lengths = batch_data[4]
            local_batch_size = len(y)
            
            fd = {self.word_ids: x,
                  self.sequence_lengths: lengths,
                  self.labels: y,
                  self.batch_size: local_batch_size,
                  self.dropout_keep_prob: 1.0}
            correct_num, probs, pred_label, loss = self.sess.run([self.correct_num, self.probs, self.pred_label, self.loss_sum], feed_dict=fd)
            total_correct += correct_num
            if(output_file != None):
                for i in range(len(y)):
                    s = ' '.join([self.config.id2word[_] for _ in filter(lambda a: a != 0, x[i].tolist())])
                    fp.write('{}, {}, {}, {}\n'.format(y[i], pred_label[i], probs[i], s))
                #fp.write('{}\n'.format(self.sess.run(self.we, feed_dict=fd)))
        if(output_file != None):
            fp.close()
            
        accuracy = total_correct * 100 / total_num
        
        return (accuracy, total_correct, total_num, loss)
