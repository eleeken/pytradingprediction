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

    DATADIR = 'data'
    TRAINDIR = 'train'


    def __init__(self):
        pass

    def prepare_trainning_file(self, datadir, fn):
        '''
        读取CSV历史行情文件，转化为训练测试数据
        :param fn:文件名
        :return:
        '''
        df = pd.read_csv(datadir + os.path.sep + fn, skipinitialspace=True, skiprows=1, names=self.COLUMNS)
        df = df.dropna(axis=0, how='any')
        df = df[(df['Adj Close'] != 0.0) & (df['Close'] != 0.0) & (df['Volume'] != 0)]
        df = df.reset_index(drop = True)
        df['Label'] = 0.0

        for i in range(len(df) - self.LABELDAYS_AHEAD):
            df.loc[i, 'Label'] = df.loc[i + self.LABELDAYS_AHEAD, 'Close']

        for i in range(len(df) - self.LABELDAYS_AHEAD, len(df)):
            df.drop(i)
        df.to_csv(self.TRAINDIR + os.path.sep + fn, index=False, sep=',')

    def prepare(self):
        """读取数据集

        Args:
            datadir: 数据文件所在目录
        Returns:
            数据集
        """

        if not os.path.exists(self.TRAINDIR):
            os.mkdir(self.TRAINDIR)

        for fn in os.listdir(self.DATADIR):
            print(">{}".format(fn))
            try:
                self.prepare_trainning_file(self.DATADIR, fn)
            except Exception as e:
                print('error!')

if __name__ == '__main__':
    dataprepare = stockdatapreparation()
    dataprepare.prepare()
    print('All finished!')
