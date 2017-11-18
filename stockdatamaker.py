from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import csv
import os
import numpy as np
import pandas as pd

from tensorflow.python.framework import dtypes


class stockdatamaker:
    """
    生成证券分析用的数据
    """
    COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    LABEL = "Pred"

    # 每行数据的交易天数
    TRAININGDATES_PER_PERIOD = 60
    # 用于计算LABEL的交易天数
    CHECKDATES_PER_PERIOD = 1
    # 每个数据的交易总天数
    DATES_PER_PERIOD = TRAININGDATES_PER_PERIOD + CHECKDATES_PER_PERIOD
    # 分类总数
    NUM_CLASSES = 4

    # 训练数据
    training_set = []
    # 标签数据
    label_set = []

    def __init__(self):
        pass

    def dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def process_label(self, csvset, r):

        close1 = float(csvset.loc[r]['Close'])
        close2 = float(csvset.loc[r + self.CHECKDATES_PER_PERIOD]['Close'])
        closegained = (close2 - close1) / close1

        label = ""
        if closegained > 0.02:
            label = 0
        elif closegained > 0.0:
            label = 1
        elif closegained > -0.02:
            label = 2
        else:
            label = 3
        return label

    def csv_to_dataset(self, fn):
        '''
        读取CSV历史行情文件，转化为训练测试数据
        :param fn:文件名
        :return:
        '''
        df = pd.read_csv(fn, skipinitialspace=True, skiprows=1, names=self.COLUMNS)
        df = df.dropna(axis=0, how='any')
        df = df[(df['Adj Close'] != 0.0) & (df['Close'] != 0.0) & (df['Volume'] != 0)]
        df = df.reset_index(drop = True)

        rows_csv = df['Date'].count()
        if  rows_csv < self.DATES_PER_PERIOD:
            return

        for j in range(int((rows_csv - self.DATES_PER_PERIOD) / self.DATES_PER_PERIOD)):

            oneperiod_set = []
            for i in range(j * self.DATES_PER_PERIOD + self.DATES_PER_PERIOD, j * self.DATES_PER_PERIOD + 1, -1):

                vol1 = float(df.loc[i - 1]['Volume'])
                vol2 = float(df.loc[i]['Volume'])
                volgained = (vol1 - vol2) / vol2

                open1 = float(df.loc[i - 1]['Open'])
                open2 = float(df.loc[i]['Open'])
                opengained = (open1 - open2) / open2

                close1 = float(df.loc[i - 1]['Close'])
                close2 = float(df.loc[i]['Close'])
                closegained = (close1 - close2) / close2

                high1 = float(df.loc[i - 1]['High'])
                high2 = float(df.loc[i]['High'])
                highgained = (high1 - high2) / high2

                low1 = float(df.loc[i - 1]['Low'])
                low2 = float(df.loc[i]['Low'])
                lowgained = (low1 - low2) / low2

                if closegained > 0.101 or closegained < -0.101:
                    break

                print('%d' % i + ' - ' + '{:f}'.format(volgained) \
                      + ' - ' + '{:f}'.format(opengained) \
                      + ' - ' + '{:f}'.format(closegained) \
                      + ' - ' + '{:f}'.format(highgained) \
                      + ' - ' + '{:f}'.format(lowgained))
                oneperiod_set.append(opengained)
                oneperiod_set.append(closegained)
                oneperiod_set.append(highgained)
                oneperiod_set.append(lowgained)
                oneperiod_set.append(volgained)

            print("size: %d" % len(oneperiod_set))

            if len(oneperiod_set) == self.TRAININGDATES_PER_PERIOD * 5:
                self.training_set.append(oneperiod_set)
                label = self.process_label(df, j * self.DATES_PER_PERIOD)
                self.label_set.append(label)

    def full_read_dataset(self, datadir):
        """读取数据集

        Args:
            datadir: 数据文件所在目录
        Returns:
            数据集
        """
        for fn in os.listdir(datadir):
            print(">{}".format(fn))
            self.csv_to_dataset(datadir + "\\" + fn)
            print("    len:{}".format(len(maker.label_set)))

        self.label_set = np.array(self.label_set, dtype=np.uint8)

        return (self.training_set, self.dense_to_one_hot(self.label_set, self.NUM_CLASSES))


if __name__ == '__main__':
    maker = stockdatamaker()
    (train, labels) = maker.full_read_dataset("data")

    print(len(maker.label_set))
