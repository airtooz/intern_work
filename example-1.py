'''
Distributed Tensorflow example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job-name="ps" --task_index=0 
pc-02$ python example.py --job-name="worker" --task_index=0 
pc-03$ python example.py --job-name="worker" --task_index=1 
pc-04$ python example.py --job-name="worker" --task_index=2 

More details here: ischlag.github.io
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time
import tensorflow.contrib.learn as skflow
from tensorflow.contrib import layers
import tempfile

def exponential_decay(l_rate, global_step):
    decay_step = 1000
    decay_rate = 0.5
    staircase = False
    return tf.train.exponential_decay(l_rate, global_step, decay_step, decay_rate, staircase)

# cluster specification
parameter_servers = ["10.81.103.122:2222"]
workers = [	"10.81.103.122:2223", 
			"10.81.103.124:2224"]
		#	"10.81.103.121:2225"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("logdir", "./mnist", "log directory")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, 
                         job_name=FLAGS.job_name, 
                         task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.001
training_epochs =1 ##5*550
logs_path = "./mnist"

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":

    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % (FLAGS.task_index),
        cluster=cluster)):
    #if True:
        # count the number of updates
        global_step = tf.get_variable('global_step', 
                                      [], 
                                      initializer = tf.constant_initializer(0), 
                                      trainable = False)

        # input images
        with tf.name_scope('input'):
          # None -> batch size can be any size, 784 -> flattened mnist image
          x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
          # target 10 output classes
          y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        
        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        '''with tf.name_scope("weights"):
            W1 = tf.Variable(tf.random_normal([784, 1024]))
            W2 = tf.Variable(tf.random_normal([1024, 10]))

        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([1024]))
            b2 = tf.Variable(tf.zeros([10]))

        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x,W1),b1)
            a2 = tf.nn.relu(z2)
            z3 = tf.add(tf.matmul(a2,W2),b2)
            y  = tf.nn.softmax(z3)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), reduction_indices=[1]))'''
        '''fully_connected = layers.fully_connected(x, 
                                                 1024, 
                                                 weights_regularizer=layers.l2_regularizer(0.1), 
                                                 biases_regularizer=layers.l2_regularizer(0.1), 
                                                 scope='FCL'+str(1))
        fully_connected_d = layers.dropout(fully_connected, keep_prob=0.8)
        for layer in xrange(1, 4): 
            fully_connected = layers.fully_connected(fully_connected_d, 
                                                     1024, 
                                                     weights_regularizer=layers.l2_regularizer(0.1), 
                                                     biases_regularizer=layers.l2_regularizer(0.1), 
                                                     scope='FCL'+str(layer+1))
            fully_connected_d = layers.dropout(fully_connected, keep_prob=0.8)
        y, cross_entropy = skflow.models.logistic_regression(fully_connected_d, y_, init_stddev=0.01)'''
        
        x_image = tf.reshape(x, [-1,28,28,1])
        stack1_conv1 = layers.convolution2d(x_image,
                                            64,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(0.1),
                                            biases_regularizer=layers.l2_regularizer(0.1),
                                            scope='stack1_Conv1')
        stack1_conv2 = layers.convolution2d(stack1_conv1,
                                            64,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(0.1),
                                            biases_regularizer=layers.l2_regularizer(0.1),
                                            scope='stack1_Conv2')
        stack1_pool = layers.max_pool2d(stack1_conv2,
                                        [2,2],
                                        padding='SAME',
                                        scope='stack1_Pool')
        '''stack2_conv1 = layers.convolution2d(stack1_pool,
                                            256,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(1.0),
                                            biases_regularizer=layers.l2_regularizer(1.0),
                                            scope='stack2_Conv1')
        stack2_conv2 = layers.convolution2d(stack2_conv1,
                                            256,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(1.0),
                                            biases_regularizer=layers.l2_regularizer(1.0),
                                            scope='stack2_Conv2')
        stack2_pool = layers.max_pool2d(stack2_conv2, 
                                        [2,2],
                                        padding='SAME',
                                        scope='stack2_Pool')
        stack3_conv1 = layers.convolution2d(stack2_pool,
                                            512,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(1.0),
                                            biases_regularizer=layers.l2_regularizer(1.0),
                                            scope='stack3_Conv1')
        stack3_conv2 = layers.convolution2d(stack3_conv1,
                                            512,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(1.0),
                                            biases_regularizer=layers.l2_regularizer(1.0),
                                            scope='stack3_Conv2')
        stack3_conv3 = layers.convolution2d(stack3_conv2,
                                            512,
                                            [3,3],
                                            weights_regularizer=layers.l2_regularizer(1.0),
                                            biases_regularizer=layers.l2_regularizer(1.0),
                                            scope='stack3_Conv3')
        stack3_pool = layers.max_pool2d(stack3_conv3, 
                                        [2,2],
                                        padding='SAME',
                                        scope='stack3_Pool')'''
        stack3_pool_flat = layers.flatten(stack1_pool, scope='stack3_pool_flat')
        fcl1 = layers.fully_connected(stack3_pool_flat, 
                                      512, 
                                      weights_regularizer=layers.l2_regularizer(0.1), 
                                      biases_regularizer=layers.l2_regularizer(0.1), 
                                      scope='FCL1')
        fcl1_d = layers.dropout(fcl1, keep_prob=0.5, scope='dropout1')
        fcl2 = layers.fully_connected(fcl1_d, 
                                      128, 
                                      weights_regularizer=layers.l2_regularizer(0.1), 
                                      biases_regularizer=layers.l2_regularizer(0.1), 
                                      scope='FCL2')
        fcl2_d = layers.dropout(fcl2, keep_prob=0.5, scope='dropout2')
        y, cross_entropy = skflow.models.logistic_regression(fcl2_d, y_, init_stddev=0.01)


        '''train_op = tf.contrib.layers.optimize_loss(loss=cross_entropy, 
                                                       global_step=global_step, 
                                                       learning_rate=0.001, 
                                                       optimizer='Adam', 
                                                       clip_gradients=1, 
                                                       learning_rate_decay_fn=exponential_decay)'''
        # specify optimizer
        '''with tf.name_scope('train'):
            # optimizer is an "operation" which we can execute in a session
            #grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            grad_op = tf.train.AdamOptimizer(learning_rate)
            
            ## comment
            rep_op = tf.train.SyncReplicasOptimizer(grad_op, 
                                                    replicas_to_aggregate=len(workers),
                                                    replica_id=FLAGS.task_index, 
                                                    total_num_replicas=len(workers),
                                                    use_locking=True)
            train_op = rep_op.minimize(cross_entropy, global_step=global_step)
            ## comment

            train_op = grad_op.minimize(cross_entropy, global_step=global_step)'''
        with tf.name_scope('train'):
            start_l_rate = 0.001
            decay_step = 1000
            decay_rate = 0.5
            learning_rate = tf.train.exponential_decay(start_l_rate, global_step, decay_step, decay_rate, staircase=False)
            grad_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            '''rep_op = tf.train.SyncReplicasOptimizer(grad_op, 
                                                    replicas_to_aggregate=len(workers),
                                                    replica_id=FLAGS.task_index, 
                                                    total_num_replicas=len(workers))'''
            train_op = tf.contrib.layers.optimize_loss(loss=cross_entropy, 
                                                       global_step=global_step, 
                                                       learning_rate=0.001, 
                                                       optimizer=grad_op, 
                                                       clip_gradients=1)
            #train_op = rep_op.minimize(cross_entropy, global_step=global_step)

        #if FLAGS.task_index == 0:
        '''chief_queue_runner = rep_op.get_chief_queue_runner()
        init_token_op = rep_op.get_init_tokens_op()
        clean_up_op = rep_op.get_clean_up_op()'''
        

        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        #tf.scalar_summary("cost", cross_entropy)
        #tf.scalar_summary("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session 
        #summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        print("Variables initialized ...")

    ##indent right
    #saver = tf.train.Saver()
    train_dir = './mnist'
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             #logdir=train_dir,
                             init_op=init_op,
                             global_step=global_step)
                             #recovery_wait_secs=1)
    
    sess_config = tf.ConfigProto(allow_soft_placement=True, 
                                 log_device_placement=False)
                                 #device_filters=["/job:ps/cpu:2", "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index,FLAGS.task_index) ])
    
    begin_time = time.time()
    frequency = 100
    with sv.prepare_or_wait_for_session(server.target, config=sess_config) as sess:
    #sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
    #with sv.managed_session(server.target) as sess:
    #sess = tf.Session()
        '''if True:
        sess.run(init_op)'''
        '''queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
        sv.start_queue_runners(sess, queue_runners)
        print ('Started %d queues for processing input data.' % len(queue_runners))'''
        
        # is chief
        '''if FLAGS.task_index == 0:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_token_op)'''

        # create log writer object (this will log on every machine)
        #writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
                
        # perform training cycles
        start_time = time.time()
        print('training start at '+str(start_time))
        for epoch in range(training_epochs):

            # number of batches in one epoch
            batch_count = int(mnist.train.num_examples/batch_size)

            count = 0
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                
                # perform the operations we defined earlier on batch
                '''_, cost, summary, step = sess.run(
                                                [train_op, cross_entropy, summary_op, global_step], 
                                                feed_dict={x: batch_x, y_: batch_y})'''
                _, cost, step = sess.run(
                                                [train_op, cross_entropy, global_step], 
                                                feed_dict={x: batch_x, y_: batch_y})
                #writer.add_summary(summary, step)

                count += 1
                if count % frequency == 0 or i+1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step+1), 
                                " Epoch: %2d," % (epoch+1), 
                                " Batch: %3d of %3d," % (i+1, batch_count), 
                                " Cost: %.4f," % cost, 
                                " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
                    count = 0
	test_batch_size = 1000
        test_batch_count = int(mnist.test.num_examples/test_batch_size)
	for i in range(test_batch_count):
	    test_batch_x, test_batch_y = mnist.test.next_batch(test_batch_size)
            print("Test-Accuracy: "+'{:.4f}'.format(sess.run(accuracy, feed_dict={x: test_batch_x, y_: test_batch_y})))
        print("Total Time: %3.2fs" % float(time.time() - begin_time))
        print("Final Cost: %.4f" % cost)

    #sv.stop()
    print("done")
