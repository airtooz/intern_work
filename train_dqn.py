import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import random
import numpy as np
from collections import deque

ACTIONS = 3 # buy, sell, do nothing
GAMMA = 0.99 # decay rate for past information
REPLAY_MEMORY = 50000 # number of previous transitions to remember
N_HIDDEN = 128 
EPSILON = 0.001

# Define weights
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

def trainNetwork(readout):
	a = tf.placeholder("float",[None,ACTIONS])
	y = tf.placeholder("float",[None])
	readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices=1)
  	cost = tf.reduce_mean(tf.square(y - readout_action))
  	train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)


	D = deque()
	


