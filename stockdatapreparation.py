from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import csv
import os
import numpy as np
import pandas as pd

from tensorflow.python.framework import dtypes


class stockdatapreparation:
    """
    生成证券分析用的数据
    """
    COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # 使用N天后的收盘价作为LABEL的数据
    LABELDAYS_AHEAD = 3


    def __init__(self):
        pass

    def prepare_csvfile(self, fn):
        '''
        读取CSV历史行情文件，转化为训练测试数据
        :param fn:文件名
        :return:
        '''
        df = pd.read_csv(fn, skipinitialspace=True, skiprows=1, names=self.COLUMNS)
        df = df.dropna(axis=0, how='any')
        df = df[(df['Adj Close'] != 0.0) & (df['Close'] != 0.0) & (df['Volume'] != 0)]
        df = df.reset_index(drop = True)
        df['Lable'] = 0.0

        for i in range(len(df) - self.LABELDAYS_AHEAD):
            df.iloc[i]['Label'] = df.iloc[i + self.LABELDAYS_AHEAD]['Close']

        for i in range(len(df) - self.LABELDAYS_AHEAD, len(df)):
            df.drop(i)





    def prepare(self, datadir):
        """读取数据集

        Args:
            datadir: 数据文件所在目录
        Returns:
            数据集
        """
        for fn in os.listdir(datadir):
            print(">{}".format(fn))
            self.prepare_csvfile(datadir + "\\" + fn)

if __name__ == '__main__':
    dataprepare = stockdatapreparation()
    dataprepare.prepare("data")
