# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016
@author: Rob Romijnders
"""
from __future__ import print_function
from yahoo_finance import Share
import datetime
import calendar
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from bn_class import *

     
"""Hyperparameters"""
num_filt_1 = 32     #Number of filters in first conv layer
num_filt_2 = 40     #Number of filters in second conv layer
num_filt_3 = 8     #Number of filters in third conv layer
num_fc_1 = 128       #Number of neurons in fully connected layer
max_iterations = 5000
batch_size = 100
dropout = 0.5       #Dropout rate in the fully connected layer
plot_row = 5        #How many rows do you want to plot in the visualization
regularization = 1e-4
learning_rate = 5e-4
input_norm = True   # Do you want z-score input normalization?

"""Load the data"""
# Datasets that were checked:
# "Two_Patterns" test-accuracy 0.999
# "ChlorineConcentration: test-accuracy 0.849
# Both of these accuracies are State-of-the-art as in this article
# "Multi-Scale Convolutional Neural Networks for Time Series Classification"
# Cui 2016 ArXiv

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
	if max(close[now-9:now]) == min(close[now-9:now]):
		return 0.0
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

def get_D(K,D):
        now = len(K)
        if now == 1:
                return 0.5
        elif now > 1:
                current = (float((D[len(D)-1])*(0.6667)) + float(K[now-1]*(0.3333)))
                return current

_ID = 2412
start = datetime.date(2008,8,2)
end = datetime.date(2014,8,2)


'''
UCR = True
if UCR:
  	dataset = "ChlorineConcentration"
  	datadir = 'UCR_TS_Archive_2015/'+ dataset + '/' + dataset
  	data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
  	data_test_val = np.loadtxt(datadir+'_TEST',delimiter=',')
else:
  	data_train = np.loadtxt('data_train_dummy',delimiter=',')
  	data_test_val = np.loadtxt('data_test_dummy',delimiter=',')
'''

print("Getting stock data from yahoo_finance...")
stock_data = getStock(_ID,start,end)
print("Finish fetching data!!")

close = []
#volume = []
rise_or_fall = []
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

print("We have", len(stock_data), "raw data")

for i in xrange(len(close)):
       	#volume.append(stock_data[i]['Volume'])
	if i >= 14:
        	if close[i] > close[i-5]:#+close[i-1]+close[i-2]+close[i-3]+close[i-4])/5  > close[i-5]:  # MA5 as label
        	  	rise_or_fall.append(0) # rise
    		else:
        	 	rise_or_fall.append(1) # fall (or equal)

#print(len(rise_or_fall)," ", len(close))

assert len(rise_or_fall) == len(close)-14
assert len(rise_or_fall) == len(D)-6

X = []

for i in range(len(rise_or_fall)-8):
	#minibatch = close[i:i+10]
	'''
	minibatch = []
	minibatch.append(K[i+1])
	minibatch.append(D[i+1])
	minibatch.append(RSV[i+1])
	'''
	minibatch = D[i:i+10]
	minibatch.insert(0,rise_or_fall[i+8])
	X.append(minibatch)

X = np.array(X)
print ("The data shape is: ",X.shape)

m = X.shape[0]
num_train = (m//10)*6
num_val = (m//10)*2

X_train = X[:num_train,1:]
y_train = X[:num_train,0]

rise = []
fall = []
for i in range(num_train,m):
	if int(X[i,0]) == 0: # rise
		rise.append(X[i])
	elif int(X[i,0]) == 1: # fall or equal
		fall.append(X[i])

num_rise_fall = m//10
test_data = rise[:num_rise_fall]
test_data.extend(fall[:num_rise_fall])
val_data = rise[num_rise_fall:]
val_data.extend(fall[num_rise_fall:])

test_data = np.array(test_data)
val_data = np.array(val_data)

X_val = val_data[:,1:]
y_val = val_data[:,0]
X_test = test_data[:,1:]
y_test = test_data[:,0]


print (X_val.shape)
print (X_test.shape)

N = X_train.shape[0]
Ntest = X_test.shape[0]
D = X_train.shape[1]


'''
data_test,data_val = np.split(data_test_val,2)
# Usually, the first column contains the target labels
X_train = data_train[:,1:]
X_val = data_val[:,1:]
X_test = data_test[:,1:]
N = X_train.shape[0]
Ntest = X_test.shape[0]
D = X_train.shape[1]
y_train = data_train[:,0]
y_val = data_val[:,0]
y_test = data_test[:,0]
'''
print('We have %s observations with %s dimensions'%(N,D))
# Organize the classes

num_classes = len(np.unique(y_train))
'''
base = np.min(y_train)  #Check if data is 0-based
if base != 0:
    y_train -=base
    y_val -= base
    y_test -= base

if input_norm:
    mean = np.mean(X_train,axis=0)
    variance = np.var(X_train,axis=0)
    X_train -= mean
    #The 1e-9 avoids dividing by zero
    X_train /= np.sqrt(variance)+1e-9
    X_val -= mean
    X_val /= np.sqrt(variance)+1e-9
    X_test -= mean
    X_test /= np.sqrt(variance)+1e-9



if True:  #Set true if you want to visualize the actual time-series
    f, axarr = plt.subplots(plot_row, num_classes)
    for c in np.unique(y_train):    #Loops over classes, plot as columns
        ind = np.where(y_train == c)
        ind_plot = np.random.choice(ind[0],size=plot_row)
        for n in xrange(plot_row):  #Loops over rows
            axarr[n,c].plot(X_train[ind_plot[n],:])
            # Only shops axes for bottom row and left column
            if not n == plot_row-1:
                plt.setp([axarr[n,c].get_xticklabels()], visible=False)
            if not c == 0:
                plt.setp([axarr[n,c].get_yticklabels()], visible=False)
    f.subplots_adjust(hspace=0)  #No horizontal space between subplots
    f.subplots_adjust(wspace=0)  #No vertical space between subplots
    plt.show()
'''

#Check for the input sizes
#assert (N>X_train.shape[1]), 'You are feeding a fat matrix for training, are you sure?'
#assert (Ntest>X_test.shape[1]), 'You are feeding a fat matrix for testing, are you sure?'

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

# Nodes for the input variables
x = tf.placeholder("float", shape=[None, D], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
keep_prob = tf.placeholder("float")
bn_train = tf.placeholder(tf.bool)          #Boolean value to guide batchnorm

# Define functions for initializing variables and standard layers
#For now, this seems superfluous, but in extending the code
#to many more layers, this will keep our code
#read-able
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("Reshaping_data") as scope:
  x_image = tf.reshape(x, [-1,D,1,1])


"""Build the graph"""
# ewma is the decay for which we update the moving average of the 
# mean and variance in the batch-norm layers
with tf.name_scope("Conv1") as scope:
  W_conv1 = weight_variable([4, 1, 1, num_filt_1], 'Conv_Layer_1')
  b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
  a_conv1 = conv2d(x_image, W_conv1) + b_conv1
  
with tf.name_scope('Batch_norm_conv1') as scope:
    ewma = tf.train.ExponentialMovingAverage(decay=0.99)
    bn_conv1 = ConvolutionalBatchNormalizer(num_filt_1, 0.001, ewma, True)
    update_assignments = bn_conv1.get_assigner() 
    a_conv1 = bn_conv1.normalize(a_conv1, train=True) 
    h_conv1 = tf.nn.relu(a_conv1)
  
with tf.name_scope("Conv2") as scope:
  W_conv2 = weight_variable([4, 1, num_filt_1, num_filt_2], 'Conv_Layer_2')
  b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
  a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
  
with tf.name_scope('Batch_norm_conv2') as scope:
    bn_conv2 = ConvolutionalBatchNormalizer(num_filt_2, 0.001, ewma, True)           
    update_assignments = bn_conv2.get_assigner() 
    a_conv2 = bn_conv2.normalize(a_conv2, train=True) 
    h_conv2 = tf.nn.relu(a_conv2)
    
with tf.name_scope("Conv3") as scope:
  W_conv3 = weight_variable([4, 1, num_filt_2, num_filt_3], 'Conv_Layer_3')
  b_conv3 = bias_variable([num_filt_3], 'bias_for_Conv_Layer_3')
  a_conv3 = conv2d(h_conv2, W_conv3) + b_conv3
  
with tf.name_scope('Batch_norm_conv3') as scope:
    bn_conv3 = ConvolutionalBatchNormalizer(num_filt_3, 0.001, ewma, True)           
    update_assignments = bn_conv3.get_assigner() 
    a_conv3 = bn_conv3.normalize(a_conv3, train=True) 
    h_conv3 = tf.nn.relu(a_conv3)

with tf.name_scope("Fully_Connected1") as scope:
  W_fc1 = weight_variable([D*num_filt_3, num_fc_1], 'Fully_Connected_layer_1')
  b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
  h_conv3_flat = tf.reshape(h_conv3, [-1, D*num_filt_3])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
  
with tf.name_scope("Fully_Connected2") as scope:
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  W_fc2 = tf.Variable(tf.truncated_normal([num_fc_1, num_classes], stddev=0.1),name = 'W_fc2')
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]),name = 'b_fc2')
  h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope("SoftMax") as scope:
    regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
                  tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) + 
                  tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + 
                  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc2,y_)
    cost = tf.reduce_sum(loss) / batch_size
    cost += regularization*regularizers
    loss_summ = tf.scalar_summary("cross entropy_loss", cost)
with tf.name_scope("train") as scope:
    tvars = tf.trainable_variables()
    #We clip the gradients to prevent explosion
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)
    # The following block plots for every trainable variable
    #  - Histogram of the entries of the Tensor
    #  - Histogram of the gradient over the Tensor
    #  - Histogram of the gradient-norm over the Tensor
    numel = tf.constant([[0]])
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient
      
      numel +=tf.reduce_sum(tf.size(variable))  
        
      h1 = tf.histogram_summary(variable.name, variable)
      h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
      h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
with tf.name_scope("Evaluating_accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(h_fc2,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    

#Define one op to call all summaries    
merged = tf.merge_all_summaries()

# For now, we collect performances in a Numpy array.
# In future releases, I hope TensorBoard allows for more
# flexibility in plotting
perf_collect = np.zeros((3,int(np.floor(max_iterations /100))))
EPOCH = 10
avg_acc = 0.
acc_over50 = 0.
acc_rise = 0.
for _ in range(EPOCH):
	with tf.Session() as sess:
		writer = tf.train.SummaryWriter("/home/airchen/Documents/coding/stock", sess.graph)

  		sess.run(tf.initialize_all_variables())
  
  		step = 0      # Step is a counter for filling the numpy array perf_collect
  		for i in range(max_iterations):
    			batch_ind = np.random.choice(N,batch_size,replace=False)
    
    			if i==0:
        			# Use this line to check before-and-after test accuracy
        			result = sess.run(accuracy, feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
        			acc_test_before = result
    			if i%100 == 0:
       				#Check training performance
       				result = sess.run(accuracy,feed_dict = { x: X_train, y_: y_train, keep_prob: 1.0, bn_train : False})
       				perf_collect[1,step] = result 
       	 
       				#Check validation performance
       				result = sess.run([accuracy,merged], feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0, bn_train : False})
       				acc = result[0]
       				perf_collect[0,step] = acc
	
				#Check current test acc
				result_test = sess.run([accuracy,merged], feed_dict = {x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
	 
       				#Write information to TensorBoard
       				summary_str = result[1]
       				writer.add_summary(summary_str, i)
       				writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
       				print("Validation accuracy at %s out of %s is %s. Current test acc is %s" % (i,max_iterations, acc, result_test[0]))
       				step +=1
    			sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train : True})
  		result = sess.run([accuracy,numel], feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
  		acc_test = result[0]
		if acc_test > 0.5:
			acc_over50 += 1.0
		if acc_test > acc_test_before:
			acc_rise += 1.0
		avg_acc+=acc_test
  		print('The network has %s trainable parameters'%(result[1]))
  
		print('The accuracy on the test data is %.3f, before training was %.3f' %(acc_test,acc_test_before))

print("----------Results----------")
print("Avg acc is %f" %(avg_acc/EPOCH))
print("Over 50 percent acc rate %f" %(acc_over50/EPOCH))
print("Acc rise after training %f" %(acc_rise/EPOCH))
print("----------Results----------")

"""Additional plots"""
plt.figure()
plt.plot(perf_collect[0],label='Valid accuracy')
plt.plot(perf_collect[1],label = 'Train accuracy')
plt.axis([0, step, 0, np.max(perf_collect)])
plt.legend()
plt.show()
# We can now open TensorBoard. Run the following line from your terminal
# tensorboard --logdir=/home/rob/Dropbox/ml_projetcs/CNN_tsc/log_tb# -*- coding: utf-8 -*-
