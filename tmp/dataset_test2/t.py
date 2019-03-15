#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import python_speech_features
import scipy.io.wavfile
import tensorflow as tf


def extract_mfcc(file_path):
    (rate,sig) = scipy.io.wavfile.read(file_path)
    mfcc = python_speech_features.mfcc(sig)
    dmfcc = python_speech_features.delta(mfcc, 2)
    ddmfcc = python_speech_features.delta(dmfcc, 2)
    mfcc = np.hstack([mfcc, dmfcc, ddmfcc])
    assert mfcc.shape[1] == 39
    
    return mfcc


def _bytes_feature(value):
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))




def main():
    #print(extract_mfcc('t.wav').shape)
    wav_file_list = ['t1.wav', 't2.wav']
    label_list = [b'yes', b'no']
    
    
    def serialize_example(feature):
    #def serialize_example(feature, label):
        feature = {
            'feature': _float_feature(feature)
        }
        #feature = {
        #    'label': _bytes_feature(label),
        #    'feature': _float_feature(feature),
        #}
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
        
    #def tf_serialize_example(feature, label):
    #    tf_string = tf.py_function(
    #        serialize_example, 
    #        (feature, label),  # pass these args to the above function.
    #        tf.string)      # the return type is <a href="../../api_docs/python/tf#string"><code>tf.string</code></a>.
    #    return tf.reshape(tf_string, ()) # The result is a scalar
        
    def tf_serialize_example(feature):
    #def tf_serialize_example(feature, label):
        tf_string = tf.py_function(serialize_example, [feature], tf.string)
        return tf.reshape(tf_string, ()) # The result is a scalar
        #return tf_string # The result is a scalar
        
    tfrecords_filename = 'data.tfrecord'
    #with tf.python_io.TFRecordWriter(tfrecords_filename) as writer:
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    mfccs = []
    for wav, label in zip(wav_file_list, label_list):
        mfcc = extract_mfcc(wav)
        print(mfcc[0])
        print(mfcc[-1])
        print(mfcc.shape)
        mfcc = mfcc.reshape((-1))
        mfcc = mfcc.tobytes()
        example_proto = tf.train.Example(features=tf.train.Features(feature={
                                                                    'feature': _bytes_feature(mfcc)
                                                                    }))
        writer.write(example_proto.SerializeToString())
    writer.close()
    #exit(0)
        
            
    # Create a description of the features.  
    feature_description = {
        'feature': tf.FixedLenFeature([], tf.string),
    }
    #feature_description = {
    #    'label': tf.FixedLenFeature([], tf.string, default_value=''),
    #    'feature': tf.FixedLenFeature([], tf.float32, default_value=0.0),
    #}
    
    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        #return tf.decode_raw(tf.parse_single_example(example_proto, feature_description), tf.float32)
        example = tf.parse_single_example(example_proto, feature_description)
        return tf.decode_raw(example['feature'], tf.float64)
    
    filenames = [tfrecords_filename]
    raw_dataset = tf.data.TFRecordDataset(filenames).map(_parse_function)
    raw_record = raw_dataset.make_one_shot_iterator()
    
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    x = raw_record.get_next()
    t = sess.run(x).reshape((-1, 39))
    print(t[0])
    print(t[-1])
    print(t.shape)
    x = raw_record.get_next()
    t = sess.run(x).reshape((-1, 39))
    print(t[0])
    print(t[-1])
    print(t.shape)
    
    
def main_():
    #tfrecords_filename = 'data.tfrecord'
    #with tf.python_io.TFRecordWriter(tfrecords_filename) as writer:
    #    writer.write(bytes('hello world', 'utf8'))
    #    
    #dataset = tf.data.TFRecordDataset(tfrecords_filename)
    
    # the number of observations in the dataset
    n_observations = int(1e4)
    
    # boolean feature, encoded as False or True
    feature0 = np.random.choice([False, True], n_observations)
    
    # integer feature, random from 0 .. 4
    feature1 = np.random.randint(0, 5, n_observations)
    
    # string feature
    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]
    
    # float feature, from a standard normal distribution
    feature3 = np.random.randn(n_observations)
    
    def serialize_example(feature0, feature1, feature2, feature3):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        
        feature = {
            'feature0': _int64_feature(feature0),
            'feature1': _int64_feature(feature1),
            'feature2': _bytes_feature(feature2),
            'feature3': _float_feature(feature3),
        }
        
        # Create a Features message using tf.train.Example.
        
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
        
    features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
    
    def tf_serialize_example(f0,f1,f2,f3):
        tf_string = tf.py_function(
            serialize_example, 
            (f0,f1,f2,f3),  # pass these args to the above function.
            tf.string)      # the return type is <a href="../../api_docs/python/tf#string"><code>tf.string</code></a>.
        return tf.reshape(tf_string, ()) # The result is a scalar
        
    serialized_features_dataset = features_dataset.map(tf_serialize_example)
    
    filename = 'test.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)
    
    
    # Create a description of the features.  
    feature_description = {
        'feature0': tf.FixedLenFeature([], tf.int64, default_value=0),
        'feature1': tf.FixedLenFeature([], tf.int64, default_value=0),
        'feature2': tf.FixedLenFeature([], tf.string, default_value=''),
        'feature3': tf.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    
    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, feature_description)
        
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames).map(_parse_function)
    raw_record = raw_dataset.make_one_shot_iterator()
    
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    x = raw_record.get_next()
    print(repr(sess.run(x)))
    x = raw_record.get_next()
    print(repr(sess.run(x)))
    x = raw_record.get_next()
    print(repr(sess.run(x)))



if(__name__ == '__main__'):
    main()
    