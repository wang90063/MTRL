import gym
env=[]
env.append(gym.make('Pong-ram-v0'))
env.append(gym.make('Freeway-ram-v0'))
env.append(gym.make('Qbert-ram-v0'))
import tensorflow as tf
import numpy as np
import random
TASK_NUMBER = 3
EPSILON=0.05
action_max_dim=18

action_dim = []

for n in range(TASK_NUMBER):
    action_dim.append(env[n].action_space.n)

# Build network

with tf.device('/device:GPU:0'), tf.variable_scope('Qnetwork_stat', reuse=tf.AUTO_REUSE) as scope:
    # input layer
    stateInput = tf.placeholder('float', [None, None, 128])

    # bidirectional-LSTM layer

    # cells for forward and backward

    with tf.variable_scope('forward_lstm'):
        lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
    with tf.variable_scope('backward_lstm'):
        lstm_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(128)

    (outputs, outputs_state) = tf.nn.bidirectional_dynamic_rnn(
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
    W_fw = tf.get_variable(name="W_fw", shape=[128, action_max_dim],
                           initializer=tf.contrib.layers.xavier_initializer())
    W_bw = tf.get_variable(name="W_bw", shape=[128, action_max_dim],
                           initializer=tf.contrib.layers.xavier_initializer())
    b =tf.get_variable(name="bias",initializer=tf.constant(0.01, shape=[action_max_dim]))

    Qvalue = tf.matmul(output_fw, W_fw) + tf.matmul(output_bw, W_bw) + b

    Qvalue = tf.reshape(Qvalue, [1, action_max_dim])


# Save and load the network

#save and loading the neural network
saver=tf.train.Saver()
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

checkpoint_list = tf.train.get_checkpoint_state("saved_networks").all_model_checkpoint_paths

train_stat = [[] for _ in range(TASK_NUMBER)]

for i in range(len(checkpoint_list)):

    saver.restore(session,checkpoint_list[i])

    for n in range(TASK_NUMBER):

        # Total reward for an episode
        episode_reward_stat = 0
        # The total time steps in each episode
        step_episode_stat = 1
        # Average episode reward in each episode
        episode_reward_stat_total = 0
        # Episode number
        episode_num_stat = -1

        Timestep=0

        done_stat=True

        observation_stat_list=[0]

        while Timestep<10000:

            if done_stat :
                episode_reward_stat_total += episode_reward_stat / step_episode_stat
                episode_reward_stat = 0
                step_episode_stat = 0
                episode_num_stat += 1
                observation_stat = env[n].reset()
                observation_stat_list[0]=observation_stat
                current_state = observation_stat_list

            Q = Qvalue.eval(feed_dict={stateInput: [current_state]})[0]
            action = np.zeros(action_max_dim)
            if random.random() <= EPSILON:
                action_index = random.randrange(action_dim)
                action[action_index] = 1
            else:
                action_index = np.argmax(Qvalue[:action_dim])
                action[action_index] = 1

            nextObservation_stat, reward_stat, done_stat, info_stat = env[n].step(np.argmax(action))

            episode_reward_stat += reward_stat

            step_episode_stat += 1

            # Update the state

            observation_stat_list[0] = nextObservation_stat

            current_state=observation_stat_list

            Timestep += 1

            train_stat[n].append(episode_reward_stat_total/episode_num_stat)

            print("EPOCH", i, "/ TASK", n, "/ TIMESTEP", Timestep)

with open("data/Pong.txt", 'a') as k:
    for ts in train_stat[0]:
        k.write(str(ts) + '\n')
    k.close()

with open("data/Freeway.txt", 'a') as k:
    for ts in train_stat[1]:
        k.write(str(ts) + '\n')
    k.close()

with open("data/QAbert.txt", 'a') as k:
    for ts in train_stat[2]:
        k.write(str(ts) + '\n')
    k.close()
