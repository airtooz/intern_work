from yahoo_finance import Share
from __future__ import print_function
import datetime

START =  "2011-01-01" # Data start date
_ID = 2330 # By default, TSMC (2330)

stock = Share(str(_ID)+'.TW')
today = datetime.date.today()
stock_data = stock.get_historical(START, str(today))
print("Historical data since", START,": ", len(stock_data))
stock_data.reverse()

i = 0
while( i < len(stock_data)):
    if (int(stock_data[i].get('Volume')) <= 0):
        stock_data.remove(stock_data[i])
        i = -1
    i += 1

print("Remove the datas with zero volume, total data ",len(stock_data))

K = []
D = []
util = []
for i in xrange(len(stock_data)):
        util.append(float(stock_data[i].get('Close')))
        if i >= 8:
                assert len(util) == 9

                #----RSV----            
                if max(util) == min(util):
                        RSV = 0.0
                else:
                        RSV = (util[len(util)-1] - min(util))/(max(util)-min(util))
                #----RSV----

                #----K----
                if i == 8:
                        temp_K = 0.5*0.6667 + RSV*0.3333
                        K.append(temp_K)
                else:
                        temp_K = K[-1]*0.6667 + RSV*0.3333
                        K.append(temp_K)
                #----K----

                #----D----
                if i == 8:
                        D.append(0.5*0.6667 + temp_K*0.3333)
                else:
                        D.append(D[-1]*0.6667 + temp_K*0.3333)
                #----D----
                util.pop(0)
                assert len(util) == 8

