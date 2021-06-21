#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from flask import Flask, render_template, request
import json
import os
import sys
import time
from collections import Counter
from datetime import timedelta

from flask_cors import CORS

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, process_file, build_vocab
app = Flask(__name__)
CORS(app, supports_credentials=True)

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


@app.route('/text_category', methods=['GET', 'POST'])  # 路由
def text_categorization():
    start_time = time.time()
    recv_data = request.get_data()
    json_re = json.loads(recv_data)
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)  # 生成词汇表
    categories, cat_to_id = read_category()  # 生成标签和对应的id表
    words, word_to_id = read_vocab(vocab_dir)  # 读取词汇和对应的词汇表
    config.vocab_size = len(words)  # 设置词汇表大小
    model = TextCNN(config)
    x_test = process_file(json_re['Text'], word_to_id, cat_to_id, config.seq_length)

    config1 = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config1.gpu_options.allow_growth = True
    session = tf.Session(config=config1)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    feed_dict = {
        model.input_x: x_test,
        model.keep_prob: 1.0
    }

    y = session.run(model.y_pred_cls, feed_dict=feed_dict)
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐', '彩票', '星座', '社会', '股票']
    print(categories[y]) # 输出结果

    return '123'


if __name__=='__main__':
    app.run()