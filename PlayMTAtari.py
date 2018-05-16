import gym
env=[]
env.append(gym.make('Pong-ram-v0'))
env.append(gym.make('Freeway-ram-v0'))
env.append(gym.make('Qbert-ram-v0'))
from MTRLagent import agent_ram
import numpy as np
TASK_NUMBER = 3

def playAtari():

    # Step 1: init BrainDQN

    action_max_dim=18    # Largest action dimension#

    action_dim = []

    for n in range(TASK_NUMBER):
        action_dim.append(env[n].action_space.n)

    agent = agent_ram(action_dim,action_max_dim)


    Observation_list = []

    done_list =[True for __ in range(TASK_NUMBER)]

    for n in range(TASK_NUMBER):

        if done_list[n]:
            Observation = env[n].reset()

            Observation_list.append([Observation])

    agent.setInitState(Observation_list)


    while agent.timeStep < 3000000 :

        reward_list=[]

        for n in range(TASK_NUMBER):

            if done_list[n]:

                Observation=env[n].reset()

                Observation_list[n]=[Observation]

        agent.setInitState(Observation_list)

        actions = agent.getAction()

        for n in range(TASK_NUMBER):

            nextObservation, reward, done, info = env[n].step(np.argmax(actions[n]))
            if reward>0:
                reward=1
            elif reward<0:
                reward=-1
            Observation_list[n]=[nextObservation]
            reward_list.append(reward)
            done_list[n]=done


        # 'Observation_list' is [[Observation],,], and 'Observation_store_list' is [Observation]
        agent.setPerception(Observation_list, actions, reward_list, done_list)




def main():
    playAtari()

if __name__ == '__main__':
    main()

