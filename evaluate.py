#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:02:05 2018

@author: shirhe-lyh
"""

"""Evaluate the trained CNN model.

Example Usage:
---------------
python3 evaluate.py \

    --frozen_graph_path: Path to model frozen graph.
"""

import numpy as np
import tensorflow as tf

from captcha.image import ImageCaptcha

flags = tf.app.flags
flags.DEFINE_string('frozen_graph_path', None, 'Path to model frozen graph.')
FLAGS = flags.FLAGS


def generate_captcha(text='1'):
    capt = ImageCaptcha(width=28, height=28, font_sizes=[24])
    image = capt.generate_image(text)
    image = np.array(image, dtype=np.uint8)
    return image


def main(_):
    alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J',
                 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 
                 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            inputs = model_graph.get_tensor_by_name('image_tensor:0')
            digits = model_graph.get_tensor_by_name('digits:0')
            digit_classes = tf.argmax(tf.nn.softmax(digits), axis=1)
            letters = model_graph.get_tensor_by_name('letters:0')
            letter_classes = tf.argmax(tf.nn.softmax(letters), axis=1)
            for i in range(10):
                label = np.random.randint(0, 34)
                image = generate_captcha(alphabets[label])
                image_np = np.expand_dims(image, axis=0)
                predicted_ = sess.run([digits, digit_classes,
                                       letters, letter_classes], 
                                           feed_dict={inputs: image_np})
                predicted_digits = np.round(predicted_[0], 2)
                predicted_digit_classes = predicted_[1]
                predicted_letters = np.round(predicted_[2], 2)
                predicted_letter_classes = predicted_[3]
                print(predicted_digits, '----', predicted_digit_classes)
                print(predicted_letters, '----', predicted_letter_classes)
                predicted_label = predicted_letter_classes[0] + 10
                if label < 10:
                    predicted_label = predicted_digit_classes[0]
                print(alphabets[predicted_label], ' vs ', alphabets[label])
            
            
if __name__ == '__main__':
    tf.app.run()
