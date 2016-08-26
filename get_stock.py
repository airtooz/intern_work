from yahoo_finance import Share
import datetime
import json
import numpy
import datetime
import calendar

_id=2330 
days=10# how many #days in a instance 
startdate=datetime.date(2015,9,16)
num_instance=30

def getStock(_id,start,end):#string
    stock = Share(str(_id)+'.TW')
    today = datetime.date.today() #todays date
    data = stock.get_historical(start, end)
    return data

def stockFormat(_id,start,end):
	arr=""
	s=getStock(_id,str(start),str(end))
	if(len(s)!=days+1):#discard the data that is not num of transaction days
		return 0
	if s[len(s)-1][ 'Close' ]>s[0][ 'Close' ]:#output _should_predict
		arr="1 "
	else:
		arr="0 "
	for x in xrange(len(s)-1,0,-1):#print out trend in time series
		arr+=s[ x ][ 'Close' ]+" "

	return arr


enddate=startdate+ datetime.timedelta(days=days)
output=""
for x in range(0,num_instance):
	temp=startdate
	for y in range(0,days+1):
		if (temp.weekday()==5)or(temp.weekday()==6):#from 0 to 6 mon to sun
			enddate=enddate+datetime.timedelta(days=1)
			if(enddate.weekday()==5):
				enddate=enddate+datetime.timedelta(days=2)
			elif(enddate.weekday()==6):
				enddate=enddate+ datetime.timedelta(days=1)			
		temp=temp+datetime.timedelta(days=1)
	print(startdate)
	#print(enddate)
	if (stockFormat(_id,startdate,enddate)):
		output+=stockFormat(_id,startdate,enddate)+"\n"
	#init
	startdate=enddate+datetime.timedelta(days=1)
	enddate=startdate+ datetime.timedelta(days=days)

fo=open(str(_id)+".txt","w")

fo.write(output)
fo.close()
