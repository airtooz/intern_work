import tensorflow as tf
import numpy as np
import sys
import time

def get_batch(n):
	x = np.random.random(n)
	y = np.exp(x)
	return x,y

def leaky_relu(x,alpha=0.2):
	return tf.maximum(alpha*x,x)

x_ = tf.placeholder(tf.float32, shape = [None,784])
y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.device("/job:ps/task:0"):
	W1 = tf.Variable(tf.truncated_normal([784,1024], stddev=0.01)))
	W2 = tf.Variable(tf.truncated_normal([1024,10], stddev=0.01)))
with tf.device("/job:ps/task:1"):
	b1 = tf.Variable(tf.zeros[1024])
	b2 = tf.Variable(tf.zeros[10])

with tf.device("/job:worker/task:0"):
	global_step = tf.Variable(0, name="global_step", trainable=False)
	h1 = tf.nn.relu(tf.matmul(x_,W1)+b1)
	y_ = tf.nn.softmax(tf.matmul(h1,W2)+b2)
	cost = -tf.reduce_sum(y*tf.log(y_))
	opt = tf.train.AdamOptimizer(0.01)
	trainer = opt.minimize(cost,global_step = global_step)
	

# cluster specification
parameter_servers = ["10.81.103.124:2222"]
workers = ["10.81.103.124:2223","10.81.103.124:2225"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":

	

