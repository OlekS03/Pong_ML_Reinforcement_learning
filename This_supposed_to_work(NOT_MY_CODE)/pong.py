# https://www.youtube.com/watch?v=tsWnOt2OKx8
# https://www.youtube.com/watch?v=PZ_0DiQtJg0
# https://www.youtube.com/watch?v=8XIPr1wvfc4

import gymnasium as gym
import numpy as np
import sys
import keras.models

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Convolution2D, Permute
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

from rl.core import Processor
from PIL import Image

env = gym.make('ALE/Pong-v5') # Set up the gym environment

nb_actions = env.action_space.n

IMG_SHAPE = (84,84)
WINDOW_LENGTH = 12

input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

train = True

class ImageProcessor(Processor):
    def process_observation(self, observation):
        IMG_SHAPE = (84,84)
        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE)
        img = img.convert('L')  # Convert to grayscale
        img = np.array(img)
        return img
    
    def process_state_batch(self, batch):
        processed_batch = batch / 255.0
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)
    
def build_model(input_shape, nb_actions=6):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (4, 4), strides=(2, 2), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(512, kernel_initializer='he_normal'))

    model.add(Activation('relu'))

    model.add(Dense(1024, kernel_initializer='he_normal'))

    model.add(Activation('relu'))

    model.add(Dense(nb_actions, kernel_initializer='he_normal'))

    model.add(Activation('linear'))

    return model

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 
                              attr='eps', 
                              value_max=1.0, 
                              value_min=0.1, 
                              value_test=0.05, 
                              nb_steps=1000000)

model = build_model(input_shape, nb_actions=nb_actions)

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

processor = ImageProcessor()

# Load weights from checkpoint file if it exists
checkpoint_filename = "CHECKPOINT.h5f"

checkpoint_callback = ModelIntervalCheckpoint(checkpoint_filename, interval=1000)

try:
    model.load_weights(checkpoint_filename)
    print("Loaded weights from checkpoint file.")
except:
    print("No checkpoint file found.")

dqn = DQNAgent(model=model, 
               nb_actions=nb_actions, 
               policy=policy,
               memory=memory, 
               processor=processor, 
               nb_steps_warmup=50000, 
               gamma=.99,
               target_model_update=10000, 
               train_interval=12, 
               delta_clip=1)

dqn.compile(Adam(learning_rate=0.0005), metrics=['mae'])

if train == True:
    metrics = dqn.fit(env, nb_steps=1000000, callbacks=[checkpoint_callback], log_interval=10000, visualize=False)
    dqn.test(env, nb_episodes=1, visualize=True)
    env.close()
    model.summary()

env = gym.make('ALE/Pong-v5', render_mode='human')

#dqn.test(env, nb_episodes=1, visualize=True)

observation = env.reset()

for step in range(1):
    observation = processor.process_observation(observation)
    #observation = processor.process_state_batch(observation)
    action = dqn.forward(observation)
    observation, reward, done, info = env.step(action)
    observation, reward, done, info = processor.process_step(observation, reward, done, info)
    if done:
        env.reset()
        env.close()

    dqn.backward(reward, terminal=done)

env.close()
