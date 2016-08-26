from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

ENV_NAME = 'MountainCar-v0' # Look up github for more...
EPISODE = 10000 # Total episodes to run
STEP = 300 # Step limitation in an episode
TEST = 10 # Test episode

GAMMA = 0.9 # discount factor
BUFFER_SIZE = 10000 # the replay buffer size
BATCH_SIZE = 32 # minibatch size
INITIAL_EPSILON = 0.5 # initial large, in order to explore more
FINAL_EPSILON = 0.01 # explore less, greedy most time

class DQN():
	def __init__(self,env):
		# experience replay
		self.replay_buffer = deque()
		# initialize parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n
		
		self.create_Q_network()
		self.create_training_method()

		# create session
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())

	def create_Q_network(self): # Use MLP
		# weights and biases
		W1 = tf.Variable(tf.truncated_normal([self.state_dim,20]))
		b1 = tf.Variable(tf.constant(0.01, shape = [20]))
		W2 = tf.Variable(tf.truncated_normal([20,self.action_dim]))
		b2 = tf.Variable(tf.constant(0.01, shape = [self.action_dim]))

		# Layer implementation
		self.state_input = tf.placeholder("float",[None,self.state_dim])
		hidden = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
		self.Q_value = tf.matmul(hidden,W2) + b2

	def create_training_method(self):
		self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot key vector
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
		Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0] # Unknown
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
	env = gym.make(ENV_NAME) # Initialize environment
	agent = DQN(env) # Initialize dqn agent

	for episode in xrange(EPISODE):
		state = env.reset() # reset() returns observation
		# start training
		for step in xrange(STEP):
			action = agent.egreedy_action(state) # e-greedy action for training
			next_state,reward,done,info = env.step(action)
			agent.perceive(state,action,reward,next_state,done)
			state = next_state
			if done:
				break
		if episode % 100 == 0:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(STEP):
					env.render()
					action = agent.action(state) # direct action for test
					state, reward, done, info = env.step(action)
					total_reward += reward
					if done:
						break
			avg_reward = total_reward/TEST
			print ("Episode: ", episode, " Evaluation Average Reward: ",avg_reward)
			if avg_reward >= 200:
				break

if __name__ == '__main__':
	main()
