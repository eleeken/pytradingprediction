import urllib.request as request
import urllib.error
import re
import time
import os
from pandas_datareader import data
import datetime

class stockdataspider:
    def __init__(self):
        self.STOCKLIST_URL = "http://quote.eastmoney.com/stocklist.html"
        self.YAHOODATA_URL = 'https://query1.finance.yahoo.com/v7/finance/download/{0}?period1=-252403200&period2=1510848000&interval=1d&events=history&crumb=i4KfR8A.S5F'

    def getlistpage(self):
        '''
        读取证券列表网页
        :return: 网页内容
        '''
        url = self.STOCKLIST_URL
        headers = {'User-Agent': 'Mozilla/4.0 (compatible;MSIE 5.5; Windows NT)'}
        urlreq = request.Request(url, headers=headers)
        try:
            respone = request.urlopen(urlreq)
            content = respone.read().decode("gbk")
            #print(content)

            return None, content
        except urllib.error.HTTPError as e:
            if hasattr(e, "code"):
                print(u"通迅失败，原因：", e.reason)
            return e, ""
        except urllib.error.ContentTooShortError as e:
            if hasattr(e, "code"):
                print(u"读取的内容太少，原因：", e.reason)
            return e, ""
        except urllib.error.URLError as e:
            if hasattr(e, "code"):
                print(u"网址链接不对，原因：", e.reason)
            return e, ""
        finally:
            pass

    def parse_and_savelist(self):
        '''
        分析并保存证券列表到文本文件
        :return:
        '''
        e,content = self.getlistpage()
        pattern = re.compile(r'<li>.*?<a.*?target=.*?html">(.*?)\((.*?)\).*?</a></li>', re.S)
        items = re.findall(pattern, content)

        # print("count:", len(items))
        f = open('stocklist.txt', 'w')
        for item in items:
            if item[1].startswith('600') or item[1].startswith('00'):
                f.write(item[1] + ',' + item[0] + '\n')
        f.close()

    def fetch_daytrandata_fromYahoo(self):
        '''

        :param code: 证券代码
        :return: 无
        '''
        start = datetime.datetime(1999, 1, 1)
        end = datetime.datetime(2017, 11, 16)

        file = open("stocklist.txt", "r")
        stockcodes = [line for line in file]
        stockcodes.insert(0, '000001.ss')
        stockcodes.insert(0, '399006.sz')
        for line in stockcodes:
            items = line.split(',')
            print("%s fetching...\n" % items[0])
            try:
                if not os.path.isfile('data/' + items[0] + '.csv'):
                    if (('.ss' not in line) & ('.sz' not in line)):
                        code = items[0].startswith('0') and items[0] + '.sz' or items[0] + '.ss'
                    else:
                        code = items[0]

                    sheet = data.DataReader(code, 'yahoo', start, end)
                    sheet.to_csv('data/' + items[0] + '.csv')
                    print('%s file was fetched.\n' % code)

                else:
                    print('exists.')
            except ConnectionError as e:
                print('Conneciton error!')
            except Exception as e:
                print('RemoteDataError!')
                pass

            time.sleep(1)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')

    spider = stockdataspider()
    spider.parse_and_savelist()
    spider.fetch_daytrandata_fromYahoo()