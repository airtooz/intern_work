from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from yahoo_finance import Share
import datetime

EPISODE = 10000 # Total episodes to run
STEP = 300 # Step limitation in an episode
TEST = 10 # Test episode

GAMMA = 0.9 # discount factor
BUFFER_SIZE = 10000 # the replay buffer size
BATCH_SIZE = 32 # minibatch size
INITIAL_EPSILON = 0.8 # initial large, in order to explore more
FINAL_EPSILON = 0.1 # explore less, greedy most time
DAY_LENGTH = 10 # the total days for a training data, also the dim of features
FEATURE_NUM = 5 # Currently: close price, volume, K, D, RSI9
UPDATE_FREQUENCY = 100 # target freezing, weights update frequency
START = "2015-01-01" # Data start date
_ID = 2412 # By default, TSMC (2330)

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
'''
close = []
for i in range(1,len(stock_data)):
	close.append(float(stock_data[i].get('Close'))-float(stock_data[i-1].get('Close')))
relative_close = np.zeros((len(close)-DAY_LENGTH, DAY_LENGTH),dtype = np.float)
for i in range(len(relative_close)):
	for j in range(DAY_LENGTH):
		relative_close[i,j] = float(close[i+j])
print (relative_close)

'''
'''
close = np.zeros((len(stock_data)-DAY_LENGTH, DAY_LENGTH), dtype=np.float)
for i in range(0, len(close)):
    for j in range(0, DAY_LENGTH):
        close[i,j] = float(stock_data[i+j].get('Close'))
print (close)
'''

# Five features: Close price, Volume, K, D, RSI 

data = np.zeros((len(stock_data)-8,FEATURE_NUM), dtype = np.float)
util = []
for i in xrange(len(stock_data)):
	util.append(float(stock_data[i-1].get('Close')))
	rise = 0.
	fall = 0.
	if i >= 8:
		assert len(util) == 9
		data[i-8][0] = float(stock_data[i].get('Close'))
		data[i-8][1] = float(float(stock_data[i].get('Volume'))/1000000.)
		#----RSI----
		for j in range(1,len(util)):
			if util[j] >= util[j-1]:
				rise += (util[j]-util[j-1])
			else:
				fall += (util[j-1]-util[j])
		if rise == 0 and fall == 0:
			data[i-8][2] = 0.5
		else:
			data[i-8][2] = rise/(rise+fall)
		#----RSI----

		#----RSV----		
		if max(util) == min(util):
			RSV = 0.0
		else:
			RSV = (util[len(util)-1] - min(util))/(max(util)-min(util))
		#----RSV----

		#----K----
		if i == 8:
			K = 0.5*0.6667 + RSV*0.3333
			data[i-8][3] = K
		else:
			K = data[i-9][3]*0.6667 + RSV*0.3333
			data[i-8][3] = K
		#----K----

		#----D----
		if i == 8:
			data[i-8][4] = 0.5*0.6667 + K*0.3333
		else:
			data[i-8][4] = data[i-9][4]*0.6667 + K*0.3333
		#----D----
		util.pop(0)
		assert len(util) == 8

box = np.zeros((len(data)-9,DAY_LENGTH*FEATURE_NUM), dtype = np.float)
for m in xrange(len(data)-9):
	for i in xrange(FEATURE_NUM):
		for j in xrange(DAY_LENGTH):
			box[m][i*10+j] = data[m+j][i]
print(box)

class TWStock():
	def __init__(self, stock_data):
		self.stock_data = stock_data
		self.train_data = self.stock_data[0:(len(stock_data)*4)//5]
		self.test_data = self.stock_data[(len(stock_data)*4)//5:len(stock_data)]
		self.stock_index = 0
		print("Training Data: ",len(self.train_data))
		print("Testing Data: ",len(self.test_data))

	def render(self):
		return

	def train_reset(self):
		self.stock_index = 0
		return self.train_data[self.stock_index]

	def test_reset(self):
		self.stock_index = 0
		return self.test_data[self.stock_index]
		
	# 0: observe, 1: having stock, 2: no stock
	def train_step(self,action): # for training, feed training data
		self.stock_index+=1
		action_reward = self.train_data[self.stock_index][0] - self.train_data[self.stock_index-1][0]
		#action_reward = self.train_data[self.stock_index][DAY_LENGTH-1] - self.train_data[self.stock_index][DAY_LENGTH-2]
		if action == 0:
			action_reward = 0
		elif action == 2:
			action_reward = -1*action_reward
		stock_done = False
		if(self.stock_index)>= len(self.train_data)-1:
			stock_done = True
        	else:
           		stock_done = False
		return self.train_data[self.stock_index], action_reward, stock_done, 0

	def test_step(self,action): # for testing, feed testing data
		self.stock_index+=1
		action_reward = self.train_data[self.stock_index][0] - self.train_data[self.stock_index-1][0]
		#action_reward = self.test_data[self.stock_index][DAY_LENGTH-1] - self.test_data[self.stock_index][DAY_LENGTH-2]
		if action == 0:
			action_reward = 0
		elif action == 2:
			action_reward = -1*action_reward
		stock_done = False
		if(self.stock_index)>= len(self.test_data)-1:
			stock_done = True
        	else:
           		stock_done = False
		return self.test_data[self.stock_index], action_reward, stock_done, 0

class DQN():
	def __init__(self,env):
		# experience replay
		self.replay_buffer = deque()
		# initialize parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		self.state_dim = DAY_LENGTH*FEATURE_NUM
		self.action_dim = 3

		self.total_updates = 0
		self.update_target = []
		self.last_target_layer = None
		self.last_policy_layer = None
		self.target_update_frequency = UPDATE_FREQUENCY
		
		self.create_Q_network()
		self.create_training_method()

		# create session
		#g_record = tf.Graph()
		#self.g_session = tf.InteractiveSession(graph=g_record)
		self.t_session = tf.InteractiveSession()

		#with g_record.as_default():
		self.R = tf.placeholder("float", shape = None)
		self.T = tf.placeholder("float", shape = None)
		R_summ = tf.scalar_summary(tags = "testing_reward", values = self.R)
		T_summ = tf.scalar_summary(tags = "training_reward", values = self.T)

		self.merged_summ = tf.merge_all_summaries()
		self.writer = tf.train.SummaryWriter(logdir = "/home/airchen/Documents/coding/stock", graph = self.t_session.graph)

		
		self.t_session.run(tf.initialize_all_variables())
	
	def get_summ(self):
		return self.t_session, self.merged_summ, self.R,self.T, self.writer

	def create_Q_network(self): 
		'''
		# Use MLP
		# weights and biase
		W1 = tf.Variable(tf.truncated_normal([self.state_dim,20]))
		b1 = tf.Variable(tf.constant(0.01, shape = [20]))
		W2 = tf.Variable(tf.truncated_normal([20,self.action_dim]))
		b2 = tf.Variable(tf.constant(0.01, shape = [self.action_dim]))

		# Layer implementation
		self.state_input = tf.placeholder("float",[None,self.state_dim])
		hidden = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
		self.Q_value = tf.matmul(hidden,W2) + b2
		'''
		# Target freezing parameters
		self.observation = tf.placeholder("float",[None,self.state_dim])
		self.next_observation = tf.placeholder("float",[None,self.state_dim])
		self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot key vector
		
		policy_input = None
		target_input = None
		# Use CNN
		# weights and biases
		W_fc = tf.Variable(tf.truncated_normal(shape = [1000,self.action_dim],stddev = 0.01))
		b_fc = tf.Variable(tf.constant(0.01,shape = [self.action_dim]))

		# Layer implementation
		policy_input = tf.reshape(self.observation,[-1,5,10,1])
		target_input = tf.reshape(self.next_observation,[-1,5,10,1])
		h_conv1 = self.conv_relu([5,5,1,5],policy_input, target_input,[1,1,1,1])
		
		policy_input = h_conv1[0]
		target_input = h_conv1[1]

		h_conv2 = self.conv_relu([5,2,5,20],policy_input, target_input,[1,1,1,1])

		policy_input = tf.reshape(h_conv2[0],shape = [-1,1000])
		target_input = tf.reshape(h_conv2[1],shape = [-1,1000])

		self.policy_q_layer = poilcy_input
		self.target_q_layer = target_input
		#self.Q_value = tf.matmul(hidden,W_fc) + b_fc

	def conv_relu(self,kernel_shape,policy_input,target_input,stride):
		weights =  tf.Variable(tf.truncated_normal(shape = kernel_shape,stddev = 0.01))
		biases = tf.Variable(tf.constant(0.01,shape = kernel_shape[-1]))

		activation = tf.nn.relu(tf.nn.conv2d(policy_input,weights,stride,padding = 'SAME') + biases)
		target_weights = tf.Variable(weights.initialized_value(), trainable=False)
		target_biases = tf.Variable(biases.initialized_value(), trainable=False)

		target_activation = tf.nn.relu(tf.nn.conv2d(target_input,weights,stride,padding = 'SAME') + biases)
		self.update_target.append(target_weights.assign(weights))
		self.update_target.append(target_biases.assign(biases))

		return [activation, target_activation]


	def create_training_method(self):
		self.y_input = tf.placeholder("float",[None])
		Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

	def train_Q_network(self):
		self.time_step+=1
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		# step 2: calculate y
		y_batch = []
		Q_value_batch = self.Q_value.eval(feed_dict = {self.state_input:next_state_batch})
		for i in range(0,BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA*np.max(Q_value_batch[i]))
		self.optimizer.run(feed_dict = {
			self.y_input:y_batch, 
			self.action_input:action_batch, 
			self.state_input:state_batch})

	def egreedy_action(self,state): # during training 
		Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]},session = self.t_session)[0] # Unknown
		if random.random() <= self.epsilon:
			return random.randint(0,self.action_dim-1)
		else:
			return np.argmax(Q_value)
		self.epsilon -= (Initial_EPSILON-FINAL_EPSILON)/10000
	
	def action(self,state): # during testing
		return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])
		
	def perceive(self,state,action,reward,next_state,done):
		# assign the to be made action into a one hot vector
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
		if len(self.replay_buffer) > BUFFER_SIZE:
			self.replay_buffer.popleft()
		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_Q_network()

def main():
	env = TWStock(box) # Initialize environment
	agent = DQN(env) # Initialize dqn agent
	sess,merged,R,T,writer = agent.get_summ()

	for episode in xrange(EPISODE):
		state = env.train_reset() # reset() returns observation
		# start training
		for step in xrange(STEP):
			action = agent.egreedy_action(state) # e-greedy action for training
			next_state,reward,done,info = env.train_step(action)
			agent.perceive(state,action,reward,next_state,done)
			state = next_state
			if done:
				break
		if episode % 20 == 0:
			train_reward = 0.
			for i in xrange(TEST):
				state = env.train_reset()
				action0_count = 0
				action1_count = 0
				action2_count = 0
				for j in xrange(STEP):
					env.render()
					action = agent.action(state)
					if action == 0:
						action0_count += 1
					elif action == 1:
						action1_count += 1
					elif action == 2:
						action2_count += 1
					else:
						print("Never come here!!")
					state, reward, done, info = env.train_step(action)
					train_reward += reward
					if done:
						break
				print("Action 0: ",action0_count,". Action 1: ", action1_count, ". Action 2: ", action2_count)
			print()
			avg_train_reward = train_reward/TEST
		
			total_reward = 0.
			for i in xrange(TEST):
				state = env.test_reset()
				action0_count = 0
				action1_count = 0
				action2_count = 0
				for j in xrange(STEP):
					env.render()
					action = agent.action(state) # direct action for test
					if action == 0:
						action0_count += 1
					elif action == 1:
						action1_count += 1
					elif action == 2:
						action2_count += 1
					else:
						print("Never come here!!")
					
					state, reward, done, info = env.test_step(action)
					total_reward += reward
					if done:
						break
				print("Action 0: ",action0_count,". Action 1: ", action1_count, ". Action 2: ", action2_count)
			avg_reward = total_reward/TEST
			print ("Episode: ", episode,"Training Average Reward: ",avg_train_reward, " Evaluation Average Reward: ",avg_reward)
			record = sess.run(merged, feed_dict={R:avg_reward,T:avg_train_reward})
			writer.add_summary(record, global_step = episode)
			writer.flush()
			if avg_reward >= 200:
				break

if __name__ == '__main__':
	main()
