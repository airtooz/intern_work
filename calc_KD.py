from __future__ import print_function
from yahoo_finance import Share
import datetime
import json
import sys
import calendar


def getStock(   _id = 2330, # By default, TSMC
                start = datetime.date(1997,8,2), # Start date
                end = datetime.date(2014,8,2) # End date
                ):
        stock = Share(str(_id)+'.TW')
        #today = datetime.date.today() #todays date
        data = stock.get_historical(str(start), str(end))
        return data  # string format

def get_RSV(close):
	now = len(close)
	current = (close[now-1] - min(close[now-9:now]))/(max(close[now-9:now]) - min(close[now-9:now]))
	assert (-0.001 < current) and (current < 1.0001)
	return current

def get_K(K,RSV):
	now = len(RSV)
	if now == 1:
		return 0.5
	elif now > 1:
		current = (float((K[len(K)-1])*(0.6667)) + float(RSV[now-1]*(0.3333)))
		return current
	else:
		print("Bug!!")

def get_D(K,D):
	now = len(K)
	if now == 1:
		return 0.5
	elif now > 1:
		current = (float((D[len(D)-1])*(0.6667)) + float(K[now-1]*(0.3333)))
		return current
	else:
		print("Bug!!")

_ID = 2330
start = datetime.date(1997,8,2)
end = datetime.date(2014,8,2)

f = open(str(_ID)+"_KD.txt",'w')

print("Getting stock data from yahoo_finance...")
stock_data = getStock(_ID, start, end)
print("Finish fetching data!!")

close = []
RSV = []
K = []
D = []

for i in xrange(len(stock_data)-1,-1,-1):
   	close.append(float(stock_data[i]['Close']))
	if i <= len(stock_data)-9:
		RSV.append(get_RSV(close))
		K.append(get_K(K,RSV))
		D.append(get_D(K,D))

assert len(close) == len(RSV)+8
assert len(RSV) == len(K)
assert len(K) == len(D)

f.write("Close  RSV    K    D")
f.write('\n')

for i in xrange(len(close)):
	if i < 8:
		f.write("%.2f"% float(close[i]))
		f.write('\n')
	else:
		f.write("%.2f"% float(close[i]))
		f.write(" ")
		f.write("%.2f"% float(RSV[i-8]))
		f.write(" ")
		f.write("%.2f"% float(K[i-8]))
		f.write(" ")
		f.write("%.2f"% float(D[i-8]))
		f.write('\n')

f.close()
