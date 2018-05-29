
import tensorflow as tf
import numpy as np
from collections import deque
import random

TASK_NUMBER=3
BATCH_SIZE = 32
UPDATE_TIME= 10000
GAMMA=0.95
FINAL_EPSILON = 0.1  # 0.001 # final value of epsilon
INITIAL_EPSILON = 1.0  # 0.01 # starting value of epsilon
OBSERVE = 50000.  # timesteps to observe before training
EXPLORE = 1000000.  # frames over which to anneal epsilon
REPLAY_MEMORY = 1000000  # number of previous transitions to remember


class agent_ram():

    def __init__(self,actions,max_actions):

        self.timeStep = 0

        self.epsilon = INITIAL_EPSILON

        # Each task has its own replay buffer
        # Initialize the buffers for all the tasks

        self.replaybuffer_list = []

        for _ in range(TASK_NUMBER):
            self.replaybuffer_list.append(deque())


        # maximum action dimension in atari games
        self.actions = max_actions
        # real action dimension of the specific game
        self.real_actions = actions

        # Initialize the weights

        self.stateInput,self.Qvalue,self.nets=self.buildQnetwork()

        # Initialize target network
        self.stateInputT,self.QvalueT,self.netsT=self.buildQnetwork()

        self.copyTargetQNetworkOperation =\
            [self.netsT[i].assign(self.nets[i]) for i in range(len(self.nets))]

        self.create_trainingmethod()

        #save and loading the neural network
        self.saver=tf.train.Saver(max_to_keep=100)
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
                print ("Could not find old network weights")


    def buildQnetwork(self):

        with tf.device('/device:GPU:0'), tf.variable_scope('Qnetwork', reuse=tf.AUTO_REUSE) as scope:

            # input layer
            stateInput = tf.placeholder('float',[TASK_NUMBER, None, 128])

            # bidirectional-LSTM layer

            # the sequence
            self.step_size = tf.placeholder(tf.float32, [1])

            # cells for forward and backward

            with tf.variable_scope('forward_lstm'):
               lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
            with tf.variable_scope('backward_lstm'):
               lstm_backward_cell= tf.nn.rnn_cell.BasicLSTMCell(128)

            (outputs,outputs_state)=tf.nn.bidirectional_dynamic_rnn(
                lstm_forward_cell,
                lstm_backward_cell,
                stateInput,
                dtype='float',
                # initial_state_fw=initial_lstm_state_forward_input,
                # initial_state_bw=initial_lstm_state_backward_input,
                time_major=True,
                scope=scope)

            output_fw = tf.reshape(outputs[0], [-1, 128])
            output_bw = tf.reshape(outputs[1], [-1, 128])
         # output layer
            W_fw=tf.get_variable(name="W_fw", shape=[128,self.actions],
                                  initializer=tf.contrib.layers.xavier_initializer())
            W_bw=tf.get_variable(name="W_bw", shape=[128,self.actions],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable(name="bias",initializer=tf.constant(0.01, shape=[self.actions]))

            Qvalue=tf.matmul(output_fw, W_fw) + tf.matmul(output_bw, W_bw) + b

            Qvalue = tf.reshape(Qvalue,[TASK_NUMBER, -1, self.actions])

        nets = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Qnetwork')

        return stateInput, Qvalue, nets
            # , initial_lstm_state_forward0, initial_lstm_state_forward1, initial_lstm_state_backward0,initial_lstm_state_backward1

    def create_trainingmethod(self):

        self.yInput = tf.placeholder('float',[None, BATCH_SIZE])
        self.actionInput = tf.placeholder('float', [None,BATCH_SIZE,self.actions])
        Q_Action = tf.reduce_sum(tf.multiply(self.Qvalue,self.actionInput),reduction_indices=2)
        self.loss = tf.reduce_mean(tf.square(self.yInput-Q_Action))
        self.trainStep = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss)

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory for all the tasks
        state_batch_task = []
        action_batch_task = []
        reward_batch_task = []
        nextState_batch_task = []
        done_batch_task = []

        for n in range(TASK_NUMBER):

            minibatch = random.sample(self.replaybuffer_list[n],BATCH_SIZE)

            state_batch = [data[0] for data in minibatch] # when feeded into 'feed_dict', the shape is "BATCH_SIZE*128"
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            nextState_batch = [data[3] for data in minibatch]
            done_batch = [data[4] for data in minibatch]

            state_batch_task.append(state_batch) # when feeded into 'feed_dict', the shape is "TASK_NUM*BATCH_SIZE*128"
            action_batch_task.append(action_batch)
            reward_batch_task.append(reward_batch)
            nextState_batch_task.append(nextState_batch)
            done_batch_task.append(done_batch)


        # Step 2: calculate y
        y_batch_task = []
        # output_state_fw_feedT,output_state_bw_feedT=self.reset_state(BATCH_SIZE)
        QValue_batch = self.QvalueT.eval(feed_dict={self.stateInputT: nextState_batch_task})
        for n in range(TASK_NUMBER):
            y_batch=[]
            for i in range(0, BATCH_SIZE):
                terminal = done_batch_task[n][i]
                if terminal:
                    y_batch.append(reward_batch_task[n][i])
                else:
                    y_batch.append(reward_batch_task[n][i] + GAMMA * np.max(QValue_batch[n][i]))
            y_batch_task.append(y_batch)

        self.trainStep.run(feed_dict={
            self.yInput: y_batch_task,
            self.actionInput: action_batch_task,
            self.stateInput: state_batch_task
        })

        # save network every 100000 iteration
        if self.timeStep % 50000==0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self,nextObservation,  action, reward, terminal):
        # the input parameters are all lists
        newState=nextObservation
        for n in range(TASK_NUMBER):
            self.replaybuffer_list[n].append((self.currentState[n][0],action[n],reward[n],nextObservation[n][0 ],terminal[n]))
            if len(self.replaybuffer_list[n])>REPLAY_MEMORY:
                self.replaybuffer_list[n].popleft()
            if self.timeStep>OBSERVE:
                self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        # forward-propagation batch_size=1
        # output_state_fw_feed, output_state_bw_feed=self.reset_state(1)
        Qvalue = self.session.run(self.Qvalue,feed_dict={self.stateInput: self.currentState})

        # get the "action_list" collecting all the
        action_list=[]
        for n in range(TASK_NUMBER):
            action_vec=np.zeros([self.actions])
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.real_actions[n])
                action_vec[action_index]=1
            else:
                action_index = np.argmax(Qvalue[n][0][:self.real_actions[n]])
                action_vec[action_index] = 1
            action_list.append(action_vec)


        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action_list


    def setInitState(self, observation):
        self.currentState = observation

    # def reset_state(self,batch_size):
    #     output_state_fw_feed = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([batch_size, 128]),np.zeros([batch_size, 128]))
    #     output_state_bw_feed = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([batch_size, 128]),np.zeros([batch_size, 128]))
    #     return output_state_fw_feed, output_state_bw_feed

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
