import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.data import Dataset


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.2f}'.format

california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
print(california_housing_dataframe.describe())


def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.
    
    Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
    """
    #selected_features = california_housing_dataframe[
    #    ["latitude",
    #    "longitude",
    #    "housing_median_age",
    #    "total_rooms",
    #    "total_bedrooms",
    #    "population",
    #    "households",
    #    "median_income"]]
    selected_features = california_housing_dataframe[["latitude", "median_income"]]
    processed_features = selected_features.copy()
    processed_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] / 
        california_housing_dataframe["population"])
    return processed_features
    
    
def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.
    
    Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
    A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets
    
    
# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
#training_examples = preprocess_features(california_housing_dataframe.head(200))
#training_targets = preprocess_targets(california_housing_dataframe.head(200))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
#validation_examples = preprocess_features(california_housing_dataframe.tail(2))
#validation_targets = preprocess_targets(california_housing_dataframe.tail(2))

print(training_examples)
print(training_targets)

class DataLoader():
    def __init__(self, feature, target):
        self.data_num = len(feature.index)
        rnd_idx = np.random.permutation(feature.index)
        feature = feature.reindex(rnd_idx)
        target = target.reindex(rnd_idx)
        self.feature_dim = len(feature.keys())
        for k, v in target.items():
            for x in target[k]:
                self.target_restruct = np.array([target[k]])
                self.target_restruct = np.reshape(self.target_restruct, [self.data_num, 1])
            break
        od = OrderedDict()
        for k, v in feature.items():
            od[k] = np.array(v, dtype=np.float32)
        self.feature_restruct = []
        for i in range(self.data_num):
            self.feature_restruct.append([od[x][i] for x in od])
            
    def data_generator(self, batch_size=64):
        iterations = math.ceil(self.data_num / batch_size)
        for i in range(iterations):
            index_start = i*batch_size
            index_end = (i+1)*batch_size
            yield i+1, iterations, self.target_restruct[index_start:index_end], self.feature_restruct[index_start:index_end]
            #yield i+1, iterations, np.array([[-100.0], [200.0]]), np.array([[1.0, 1.0], [2.0, 2.0]])
    

print('loading training data ...')
train_dl = DataLoader(training_examples, training_targets)
print('loading valid data ...')
valid_dl = DataLoader(validation_examples, validation_targets)
print('load data end')
#for x in train_dl.data_generator():
#    print(x)
#exit(0)
feature_dim = train_dl.feature_dim
print('feature_dim = {}'.format(feature_dim))

feature = tf.placeholder(dtype=tf.float32, shape=(None, feature_dim), name='feature')
target = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='target')

#logits = tf.contrib.layers.fully_connected(feature, 1)
layer_dense1 = tf.layers.Dense(50, kernel_initializer=tf.random_normal_initializer(), bias_initializer=tf.random_normal_initializer())
layer_dense2 = tf.layers.Dense(50, kernel_initializer=tf.random_normal_initializer(), bias_initializer=tf.random_normal_initializer())
dense_output = layer_dense1(feature)
logits = tf.contrib.layers.fully_connected(dense_output, 1, activation_fn=None)

sb = tf.subtract(target, logits)
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, target))))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
train_step = optimizer.minimize(loss)

#sess_config = tf.ConfigProto()
#sess_config.gpu_options.allow_growth = True
#with tf.Session(config=sess_config) as sess:
#    for y, x in train_dl.data_generator():
#        logit = sess.run([logits], feed_dict={feature: x, target: y})
#        print(logit)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
saver = tf.train.Saver()
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        epoch += 1
        epoch_loss = 0
        valid_loss = 0
        #pbar = tqdm(total=math.ceil(train_dl.data_num/64), desc='Epoch %d training' %(epoch))
        for iter, total_iter, y, x in train_dl.data_generator():
            #pbar.update(1)
            _, sess_train_loss, lg = sess.run([train_step, loss, logits], feed_dict={feature: x, target: y})
            epoch_loss += sess_train_loss/len(y)
            #print('x = {}'.format(x))
            #print('y = {}, \nlg = {}'.format(y, lg))
        #pbar.close()
        print('epoch {} training loss = {}'.format(epoch, epoch_loss, lg))
        #pbar = tqdm(total=math.ceil(valid_dl.data_num/64), desc='Epoch %d valid' %(epoch))
        for iter, total_iter, y_, x_ in valid_dl.data_generator():
            #pbar.update(1)
            sess_valid_loss, lg_ = sess.run([loss, logits], feed_dict={feature: x_, target: y_})
            valid_loss += sess_valid_loss/len(y_)
            #print('x_ = {}'.format(x_))
            #print('y_ = {}, \nlg_ = {}'.format(y_, lg_))
        #pbar.close()
        print('epoch {} valid loss = {}'.format(epoch, valid_loss))
        #print('')
        print('===========================================')
    print('x = {}'.format(x))
    print('y = {}, \nlg = {}'.format(y, lg))
    
