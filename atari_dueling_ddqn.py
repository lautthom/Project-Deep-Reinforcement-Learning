import numpy as np
import gym
import torch
import random
from collections import deque
import matplotlib.pyplot as plt
import argparse

from gym.wrappers import AtariPreprocessing, FrameStack


class NeuralNetwork(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.ReLU = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1)
        self.flatten = torch.nn.Flatten()
        self.dense_state1 = torch.nn.Linear(in_features=3136, out_features=512)
        self.dense_state2 = torch.nn.Linear(512, 1)
        self.dense_advantage1 = torch.nn.Linear(3136, 512)
        self.dense_advantage2 = torch.nn.Linear(512, output_size)
        self.output_layer = torch.nn.Linear(512, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.flatten(x)
        y = self.dense_state1(x)
        y = self.ReLU(y)
        y = self.dense_state2(y)
        x = self.dense_advantage1(x)
        x = self.ReLU(x)
        x = self.dense_advantage2(x)
        return y + (x - torch.mean(x))


class DQNLearner:

    def __init__(self, dim_observations, dim_actions):
        self.epsilon = INITIAL_EXPLORATION
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.model = NeuralNetwork(dim_observations, dim_actions).to(device)
        self.target_net = NeuralNetwork(dim_observations, dim_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_action(self, observation, training=True):
        if random.random() < self.epsilon and training:
            return env.action_space.sample()
        else:
            observation = torch.from_numpy(observation).to(device).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(self.model(observation)).item()
            return action

    def add_memory(self, last_observation, action, reward, observation, done):
        self.memory.append([last_observation, action, reward, observation, done])

    def learn(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        last_observations, actions, rewards, observations, dones = unpack_batch(batch)
        actions = actions.type(torch.int64).unsqueeze(1)

        q_values = self.model(last_observations).gather(1, actions)
        with torch.no_grad():
            best_actions = torch.argmax(self.model(observations), dim=1)
            best_actions = best_actions.type(torch.int64).unsqueeze(1)
            next_q_values = self.target_net(observations).gather(1, best_actions)
        next_q_values = torch.where(dones, torch.zeros(1, 1).to(device), next_q_values.squeeze(1))
        target = (next_q_values * DISCOUNT_FACTOR) + rewards
        target = torch.reshape(target, (BATCH_SIZE, 1))

        loss_fn = torch.nn.HuberLoss()
        loss = loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def update_target_network(self):
        self.target_net.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        if self.epsilon <= FINAL_EXPLORATION:
            return
        self.epsilon -= EXPLORATION_DECAY


def unpack_batch(batch):
    last_observations = [i[0] for i in batch]
    last_observations = torch.from_numpy(np.array(last_observations)).to(device)
    actions = [i[1] for i in batch]
    actions = torch.from_numpy(np.array(actions)).to(device)
    rewards = [i[2] for i in batch]
    rewards = torch.from_numpy(np.array(rewards)).to(device)
    observations = [i[3] for i in batch]
    observations = torch.from_numpy(np.array(observations)).to(device)
    dones = [i[4] for i in batch]
    dones = torch.from_numpy(np.array(dones)).to(device)
    return last_observations, actions, rewards, observations, dones


def transform_observation(observation):
    return np.array(observation)


def reward_clipping(reward):
    return 1 if reward > 0 else -1 if reward < 0 else 0


def run_evaluation(episode_evaluation):
    observation = env.reset()[0]
    observation = transform_observation(observation)
    reward_episode = 0
    duration_episode = 0
    while True:
        action = learner.get_action(observation, training=False)
        observation, reward, done, truncation, _ = env.step(action)
        observation = transform_observation(observation)
        reward = reward_clipping(reward)
        reward_episode += reward
        duration_episode += 1

        if done or truncation:
            print(f"Episode: {episode_evaluation}, Reward: {reward_episode}")
            return reward_episode


def plot_results(rewards, losses):
    mean_rewards = []
    mean_losses = []
    for i in range(len(rewards) - 99):
        mean_rewards.append(sum(rewards[i:100+i]) / 100)
    for i in range(len(losses) - 99):
        mean_losses.append(sum(losses[i:100+i]) / 100)

    plt.figure()
    plt.suptitle("Results DDQN")
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.plot(range(99, len(mean_rewards) + 99), mean_rewards, color='orange')
    plt.ylabel('Rewards')

    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.plot(range(99, len(mean_losses) + 99), mean_losses, color='orange')
    plt.ylabel('Losses')
    plt.xlabel('Episode')

    plt.show()


parser = argparse.ArgumentParser(description="Run DQN or one of its extensions on Atari games")
parser.add_argument("-t", "--train", action="store_true", help="if argument is given, new agent will be trained,"
                                                               " otherwise already trained agent will be loaded")
parser.add_argument("-dqn", "--dqn", action="store_false",
                    help="if argument is given, DQN will be used as algorithm, otherwise DDQN will be used")
parser.add_argument("-sn", "--single_network", action="store_false",
                    help="if argument is given, single stream network is used, otherwise dueling network is used")
parser.add_argument("-g", "--game", default="pong",
                    help='choose game that is played; default game is "Pong"')
parser.add_argument("-tf", "--training_frames", default=1_000_000, type=int,
                    help="choose the number of frames, that is used for training; default is 1,000,000 frames")
parser.add_argument("-b", "--batch_size", default=128, type=int,
                    help="choose the batch size, that is used for training; default is 128 frames")
parser.add_argument("-r", "--replay_size", default=100_000, type=int,
                    help="choose the replay memory size, that is used for training; default is 100,000")
parser.add_argument("-tu", "--target_update", default=1_000, type=int,
                    help="choose the frequency, with which the target network is updated; default is every 1,000 frames"
                    )
parser.add_argument("-u", "--update_frequency", default=16, type=int,
                    help="choose the frequency, with which the policy network is updated; default is every 16 frames")
parser.add_argument("-l", "--learning_rate", default=2e-4, type=float,
                    help="choose the learning rate, that is used for training; default is 0.002")
parser.add_argument("-ie", "--initial_exploration", default=1, type=float,
                    help="choose the initial exploration rate; default is 1")
parser.add_argument("-fe", "--final_exploration", default=0.02, type=float,
                    help="choose the initial exploration rate; default is 1")
parser.add_argument("-f", "--final_exploration_frame", type=int,
                    help="choose the final exploration frame, after which the final exploration rate is used;"
                         " default is a 10th of total training frames")

args = parser.parse_args()
TRAINING_FRAMES = args.training_frames
BATCH_SIZE = args.batch_size
REPLAY_MEMORY_SIZE = args.replay_size
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = args.target_update
DISCOUNT_FACTOR = 0.99
UPDATE_FREQUENCY = args.update_frequency
LEARNING_RATE = args.learning_rate
INITIAL_EXPLORATION = args.initial_exploration
FINAL_EXPLORATION = args.final_exploration
if args.final_exploration_frame:
    FINAL_EXPLORATION_FRAME = args.final_exploration_frame
else:
    FINAL_EXPLORATION_FRAME = TRAINING_FRAMES / 10
EXPLORATION_DECAY = (INITIAL_EXPLORATION - FINAL_EXPLORATION) \
                    / (FINAL_EXPLORATION_FRAME / TARGET_NETWORK_UPDATE_FREQUENCY)
REPLAY_START_SIZE = 10_000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
env = gym.make('ALE/Pong-v5',
               frameskip=1,  # no frameskip, as frameskip is applied in the AtariPreprocessing-wrapper
               repeat_action_probability=0,  # repeat_action_probability set to 0 because not applied in original paper
               full_action_space=False)
env = AtariPreprocessing(env, scale_obs=True)
env = FrameStack(env, AGENT_HISTORY_LENGTH)
number_valid_actions = env.action_space.n
learner = DQNLearner(observation.shape[0], number_valid_actions)
learner.update_target_network()

observation = env.reset()[0]
observation = transform_observation(observation)

number_current_frame, trained_episode, duration_episode, rewards_episode = 0, 0, 0, 0
losses_episode, rewards, losses = [], [], []
while number_current_frame < TRAINING_FRAMES:
    action = learner.get_action(observation)
    last_observation = observation

    observation, reward, done, truncation, info = env.step(action)
    observation = transform_observation(observation)
    reward = reward_clipping(reward)
    rewards_episode += reward

    learner.add_memory(last_observation, action, reward, observation, done)

    if number_current_frame % UPDATE_FREQUENCY == 0 and number_current_frame > REPLAY_START_SIZE:
        loss = learner.learn()
        losses_episode.append(loss.item())

    if number_current_frame % TARGET_NETWORK_UPDATE_FREQUENCY == 0 and number_current_frame > REPLAY_START_SIZE:
        learner.update_target_network()
        learner.update_epsilon()

    number_current_frame += 1
    duration_episode += 1
    if done or truncation:
        trained_episode += 1
        print(f"Frame Number: {number_current_frame}, Episode: {trained_episode}, "
              f"Duration: {duration_episode}, Reward: {rewards_episode}")
        rewards.append(rewards_episode)
        if losses_episode:
            losses.append(sum(losses_episode) / len(losses_episode))
        else:
            losses.append(0)
        duration_episode, rewards_episode = 0, 0
        losses_episode.clear()

        observation = env.reset()[0]
        observation = transform_observation(observation)

print('Training complete')
plot_results(rewards, losses)

rewards.clear()
for i in range(100):
    rewards.append(run_evaluation(i + 1))
print(sum(rewards) / len(rewards))
env.close()