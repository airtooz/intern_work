from yahoo_finance import Share
import datetime
import json
import numpy
import datetime
import calendar

RNN_SIZE = 128 # hidden layer unit numbers
NUM_LAYERS = 3 # number of layers of LSTM

def getStock(	_id = 2330, # TSMC
		start = datetime.date(2010,8,2), # Start date
		end = datetime.date(2013,8,2) # End date
		):
    	stock = Share(str(_id)+'.TW')
    	#today = datetime.date.today() #todays date
    	data = stock.get_historical(str(start), str(end))
    	return data  # string format

class Model():
	def __init__(self):
		cell = rnn_cell.BasicLSTMCell(RNN_SIZE)
		self.cell = cell = rnn_cell.MultiRNNCell([cell]*NUM_LAYERS)




