#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:54:02 2018

@author: shirhe-lyh
"""

import tensorflow as tf

from abc import ABCMeta
from abc import abstractmethod

slim = tf.contrib.slim


class BaseModel(object):
    """Abstract base class for any model."""
    __metaclass__ = ABCMeta
    
    def __init__(self, num_classes_dict):
        """Constructor.
        
        Args:
            num_classes: Number of classes.
        """
        self._num_classes_dict = num_classes_dict
        
    @property
    def num_classes_dict(self):
        return self._num_classes_dict
    
    @abstractmethod
    def preprocess(self, inputs):
        """Input preprocessing. To be override by implementations.
        
        Args:
            inputs: A float32 tensor with shape [batch_size, height, width,
                num_channels] representing a batch of images.
            
        Returns:
            preprocessed_inputs: A float32 tensor with shape [batch_size, 
                height, widht, num_channels] representing a batch of images.
        """
        pass
    
    @abstractmethod
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        pass
    
    @abstractmethod
    def postprocess(self, prediction_dict, **params):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        pass
    
    @abstractmethod
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        pass
    
        
class Model(BaseModel):
    """xxx definition."""
    
    def __init__(self,
                 is_training,
                 num_classes_dict={'digits': 10, 'letters': 24}):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        super(Model, self).__init__(num_classes_dict=num_classes_dict)
        
        self._is_training = is_training
        self._class_order = ['digits', 'letters']
        
    def preprocess(self, inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return preprocessed_inputs
    
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        net = preprocessed_inputs
        net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3')
        net = slim.flatten(net, scope='flatten')
        net = slim.dropout(net, keep_prob=0.5,
                           is_training=self._is_training)
        net = slim.fully_connected(net, 512, scope='fc1')
        net = slim.fully_connected(net, 512, scope='fc2')
        prediction_dict = {}
        for class_name, num_classes in self.num_classes_dict.items():
            logits = slim.fully_connected(net, num_outputs=num_classes, 
                                          activation_fn=None, 
                                          scope='Predict/' + class_name)
            prediction_dict[class_name] = logits
        return prediction_dict
    
    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        postprecessed_dict = {}
        for class_name in self.num_classes_dict:
            logits = prediction_dict[class_name]
#            logits = tf.nn.softmax(logits, name=class_name)
            postprecessed_dict[class_name] = logits
        return postprecessed_dict
    
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists: A list of tensors holding groundtruth
                information, with one entry for each task.
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        onehot_labels_dict = self._onehot_groundtruth_dict(groundtruth_lists)
        for class_name in self.num_classes_dict:
            weights = tf.cast(tf.greater(
                tf.reduce_sum(onehot_labels_dict[class_name], axis=1), 0),
                dtype=tf.float32)
            slim.losses.softmax_cross_entropy(
                logits=prediction_dict[class_name], 
                onehot_labels=onehot_labels_dict[class_name],
                weights=weights,
                scope='Loss/' + class_name)
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict
    
    def _onehot_groundtruth_dict(self, groundtruth_lists):
        """Transform groundtruth lables to one-hot formats.
        
        Args:
            groundtruth_lists: A dict of tensors holding groundtruth
                information, with one entry for task.
                
        Returns:
            onehot_labels_dict: A dictionary mapping strings (class names) 
                to one-hot lable tensors.
        """
        one_hot = tf.one_hot(
            groundtruth_lists, depth=sum(self.num_classes_dict.values()))
        onehot_labels_dict = {}
        start_index = 0
        for class_name in self._class_order:
            onehot_labels_dict[class_name] = tf.slice(
                one_hot, [0, start_index], 
                [-1, self.num_classes_dict[class_name]])
            start_index += self.num_classes_dict[class_name]
        return onehot_labels_dict
    
    def accuracy(self, postprocessed_dict, groundtruth_lists):
        """Calculate accuracy.
        
        Args:
            postprocessed_dict: A dictionary containing the postprocessed 
                results
            groundtruth_lists: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            accuracy: The scalar accuracy.
        """
        onehot_labels_dict = self._onehot_groundtruth_dict(groundtruth_lists)
        num_corrections = 0.
        for class_name in self.num_classes_dict:
            predicted_argmax = tf.argmax(tf.nn.softmax(
                postprocessed_dict[class_name]), axis=1)
            onehot_predicted = tf.one_hot(
                predicted_argmax, depth=self.num_classes_dict[class_name])
            onehot_sum = tf.add(onehot_labels_dict[class_name],
                                onehot_predicted)
            correct = tf.greater(onehot_sum, 1)
            num = tf.reduce_sum(tf.cast(correct, tf.float32))
            num_corrections += num
        total_nums = tf.cast(tf.shape(groundtruth_lists)[0], dtype=tf.float32)
        accuracy = num_corrections / total_nums
        return accuracy 
        
    
