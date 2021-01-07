import os
from random import random
from time import sleep
from os import system, name
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import clear_output

# SOURCE : https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

class Qlearning:

    def random(self, env):  # This function makes random moves till the episodes ends
        epochs = 0
        penalties, reward = 0, 0

        print("Using the Random Policy")

        frames = []  # for animation

        done = False

        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            # Put each rendered frame into dict for animation
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            }
            )

            epochs += 1

        print("Timesteps taken: {}".format(epochs))
        print("Penalties incurred: {}".format(penalties))

    def print_frames(self, frames):
        for i, frame in enumerate(frames):
            os.system('cls')
            clear_output(wait=True)
            print(frame['frame'])
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            sleep(.1)

    def train(self, env, q_table): #This fuction updates the Q table using Bellman's Equation

        #These are the hyperparameters
        alpha = .1
        gamma = .6
        epsilon = .1

        #For plotting metrics
        all_epochs = []
        all_penalties = []
        total_epsiodes= []

        print("Training Started")
        for i in range(1, 100001):
            state = env.reset()

            epochs, penalties, reward, = 0, 0, 0
            done = False

            while not done:
                if np.random.uniform(0,1) < epsilon:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(q_table[state])  # Exploit learned values

                next_state, reward, done, info = env.step(action)

                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) #The sate value function which updates the Q value
                q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            all_epochs.append(epochs)
            all_penalties.append(penalties)
            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")

                total_epsiodes.append(i)

        all_epochs = all_epochs[:1000]
        all_penalties = all_penalties[:1000]


        #Plotting the STEPS vs EPISODES graph
        sns.lineplot(total_epsiodes, all_epochs)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.show()

        # Plotting the PENALTIES vs EPISODES graph
        sns.lineplot(total_epsiodes, all_penalties)
        plt.xlabel("Episode")
        plt.ylabel("penalties")
        plt.show()



        print("Training finished.\n")



    def test(self, env, q_table):
        total_epochs, total_penalties = 0, 0
        episodes = 100
        print("Using the optimal policy")

        for _ in range(episodes):
            state = env.reset()
            epochs, penalties, reward = 0, 0, 0

            done = False

            while not done:


                action = np.argmax(q_table[state])

                state, reward, done, info = env.step(action)

                if reward == -10:
                    penalties += 1

                epochs += 1

            total_penalties += penalties
            total_epochs += epochs

        print(f"Results after {episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / episodes}")
        print(f"Average penalties per episode: {total_penalties / episodes}")


if __name__ == "__main__":


    taxi = gym.make("Taxi-v3").env
    q_table = np.zeros([taxi.observation_space.n, taxi.action_space.n])
    test = Qlearning()

    taxi.render()

    test.random(taxi)
    taxi.render()

    test.train(taxi, q_table)
    taxi.reset()


    taxi.render()
    test.test(taxi, q_table)
    taxi.render()


