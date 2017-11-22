#coding=gbk

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class stockpredictlstm:

    RNN_UNIT = 10
    INPUTSIZE = 6
    OUTPUTSIZE = 1
    LR = 0.0006

    COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", 'Label']
    LABEL = 'Label'
    CLIDX_LABEL = len(COLUMNS) - 2  # the column index (start from 0) of Label column after removing unused columns.
    CLIDX_BEGIN = 1
    CLIDX_END = len(COLUMNS)

    weights = {
        'in': tf.Variable(tf.random_normal([INPUTSIZE, RNN_UNIT])),
        'out': tf.Variable(tf.random_normal([RNN_UNIT, 1]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[RNN_UNIT, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }

    def __init__(self, fn):
        df = pd.read_csv(fn, skipinitialspace=True, skiprows=1, names=self.COLUMNS)
        #df = df.dropna(axis=0, how='any')
        df = df[df[self.LABEL] != 0]
        df = df.reset_index(drop = True)
        self.data = df.iloc[:, self.CLIDX_BEGIN:self.CLIDX_END].values
        #self.data = df.iloc[:, 2:10].values

    @property
    def datalen(self):
        return len(self.data)

    def get_train_data(self, batch_size=60, time_step=20, train_begin=0, train_end=1800):
        batch_index = []
        data_train = self.data[train_begin:train_end]
        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)
        train_x, train_y = [], []
        for i in range(len(normalized_train_data) - time_step):
            if i % batch_size == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i + time_step, : self.CLIDX_LABEL]
            y = normalized_train_data[i:i + time_step, self.CLIDX_LABEL, np.newaxis]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data) - time_step))
        return batch_index, train_x, train_y

    def get_test_data(self, time_step=20, test_begin=1800):
        data_test = self.data[test_begin:]
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test - mean) / std
        size = (len(normalized_test_data) + time_step - 1) // time_step
        test_x, test_y = [], []
        for i in range(size - 1):
            x = normalized_test_data[i * time_step:(i + 1) * time_step, : self.CLIDX_LABEL]
            y = normalized_test_data[i * time_step:(i + 1) * time_step, self.CLIDX_LABEL]
            test_x.append(x.tolist())
            test_y.extend(y)
        test_x.append((normalized_test_data[(i + 1) * time_step:, : self.CLIDX_LABEL]).tolist())
        test_y.extend((normalized_test_data[(i + 1) * time_step:, self.CLIDX_LABEL]).tolist())
        return mean, std, test_x, test_y

    def LSTM(self, X):
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = self.weights['in']
        b_in = self.biases['in']
        input = tf.reshape(X, [-1, self.INPUTSIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, self.RNN_UNIT])  # 将tensor转成3维，作为lstm cell的输入
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.RNN_UNIT)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, self.RNN_UNIT])
        w_out = self.weights['out']
        b_out = self.biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states

    '''
    def train_lstm(self, batch_size=80, time_step=15, train_begin=2000, train_end=5800):
        X = tf.placeholder(tf.float32, shape=[None, time_step, self.INPUTSIZE])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, self.OUTPUTSIZE])
        batch_index, train_x, train_y = self.get_train_data(batch_size, time_step, train_begin, train_end)
        pred, _ = self.LSTM(X)

        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        train_op = tf.train.AdamOptimizer(self.LR).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        module_file = tf.train.latest_checkpoint('model')
        with tf.Session() as sess:
            if module_file != None:
                saver.restore(sess, module_file)

            for i in range(2000):
                for step in range(len(batch_index) - 1):
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                   Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print(i, loss_)
                if i % 200 == 0:
                    print('save model...', saver.save(sess, 'stockpredlstm.model', global_step=i))
    '''

    def train_lstm(self, batch_size=60, time_step=20, train_begin=0, train_end=1800):
        X = tf.placeholder(tf.float32, shape=[None, time_step, self.INPUTSIZE])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, self.OUTPUTSIZE])
        batch_index, train_x, train_y = self.get_train_data(batch_size, time_step, train_begin, train_end)
        with tf.variable_scope("sec_lstm"):
            pred, _ = self.LSTM(X)
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        train_op = tf.train.AdamOptimizer(self.LR).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(50):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
                for step in range(len(batch_index) - 1):
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                   Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print("Number of iterations:", i, " loss:", loss_)
            print("model_save: ", saver.save(sess, 'model' + os.path.sep + 'modle.ckpt'))
            # 我是在window下跑的，这个地址是存放模型的地方，模型参数文件名为modle.ckpt
            # 在Linux下面用 'model_save2/modle.ckpt'
            print("The train has finished")

    '''
    def prediction(self, time_step=20):
        X = tf.placeholder(tf.float32, shape=[None, time_step, self.INPUTSIZE])
        # Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
        mean, std, test_x, test_y = self.get_test_data(time_step)
        pred, _ = self.LSTM(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint()
            saver.restore(sess, module_file)
            test_predict = []
            for step in range(len(test_x) - 1):
                prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                predict = prob.reshape((-1))
                test_predict.extend(predict)
            test_y = np.array(test_y) * std[7] + mean[7]
            test_predict = np.array(test_predict) * std[7] + mean[7]
            acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])

            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b')
            plt.plot(list(range(len(test_y))), test_y, color='r')
            plt.show()
    '''

    def prediction(self, time_step=20, test_begin=1800):
        X = tf.placeholder(tf.float32, shape=[None, time_step, self.INPUTSIZE])
        mean, std, test_x, test_y = self.get_test_data(time_step, test_begin)
        with tf.variable_scope("sec_lstm", reuse=True):
            pred, _ = self.LSTM(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint('model_save2')
            saver.restore(sess, module_file)
            test_predict = []
            for step in range(len(test_x) - 1):
                prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                predict = prob.reshape((-1))
                test_predict.extend(predict)
            test_y = np.array(test_y) * std[self.CLIDX_LABEL] + mean[self.CLIDX_LABEL]
            test_predict = np.array(test_predict) * std[self.CLIDX_LABEL] + mean[self.CLIDX_LABEL]
            acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
            print("The accuracy of this predict:", acc * 100, '%')
            # 以折线图表示结果
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b', )
            plt.plot(list(range(len(test_y))), test_y, color='r')
            plt.show()

if __name__ == '__main__':
    predict = stockpredictlstm('train' + os.path.sep + '600118.csv')
    print(predict.datalen)
    predict.train_lstm(batch_size=60, time_step=20, train_begin=0, train_end=predict.datalen - 500)
    predict.prediction(time_step=20, test_begin=predict.datalen - 500)
