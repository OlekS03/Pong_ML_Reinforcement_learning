import os
import time
import random
import argparse
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
from ale_py import ALEInterface, roms
from datetime import datetime

# HYPERPARAMETERS
GAMMA = 0.99
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99995
LEARNING_RATE = 0.00025
MEMORY_SIZE = 50000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 1000
MODEL_FILE = "pong_model.weights.h5"
DATA_FILE = "training_data.json"

# UTILITIES
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[34:194, :]
    resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
    return (resized / 255.0).astype(np.float32)

def build_model(action_size):
    model = models.Sequential([
        layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=(84, 84, 1)),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )
    return model

def save_progress(model, epsilon, episode):
    model.save_weights(MODEL_FILE)
    with open(DATA_FILE, 'w') as f:
        json.dump({'epsilon': epsilon, 'episode': episode}, f)
    print(f"\n--- Progress Saved (Epsilon: {epsilon:.4f}, Episode: {episode}) ---")

def load_progress(model):
    epsilon = 1.0
    start_episode = 0
    if os.path.exists(MODEL_FILE):
        model.load_weights(MODEL_FILE)
        print("Loaded weights from disk.")
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            epsilon = data['epsilon']
            start_episode = data['episode']
        print(f"Resuming from Episode {start_episode} with Epsilon {epsilon:.4f}")
    return epsilon, start_episode

# TEST MODE
def run_test(rom_name="pong"):
    print("\nRUNNING TEST MODE (NO RANDOMNESS, NO TRAINING)")

    ale = ALEInterface()
    ale.loadROM(roms.get_rom_path(rom_name))

    action_set = ale.getMinimalActionSet()
    action_size = len(action_set)

    model = build_model(action_size)
    model.load_weights(MODEL_FILE)

    ale.reset_game()
    state = preprocess_frame(ale.getScreenRGB())
    state = np.reshape(state, (1, 84, 84, 1))

    total_reward = 0

    while not ale.game_over():
        # Always choose best action (epsilon = 0)
        q_values = model(state, training=False)
        action_idx = np.argmax(q_values[0])
        action = action_set[action_idx]
        print(action)

        reward = ale.act(action)
        total_reward += reward

        next_raw = ale.getScreenRGB()
        next_state = preprocess_frame(next_raw)
        state = np.reshape(next_state, (1, 84, 84, 1))

        display_img = cv2.resize(next_raw, (400, 400))
        cv2.imshow("Pong AI - TEST MODE", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    print(f"\nTEST COMPLETE | Total Reward: {total_reward}")
    cv2.destroyAllWindows()

# TRAINING LOOP
def run_training(rom_name="pong"):
    ale = ALEInterface()
    ale.loadROM(roms.get_rom_path(rom_name))

    action_set = ale.getMinimalActionSet()
    action_size = len(action_set)

    main_model = build_model(action_size)
    target_model = build_model(action_size)

    epsilon, start_episode = load_progress(main_model)
    target_model.set_weights(main_model.get_weights())

    memory = deque(maxlen=MEMORY_SIZE)
    total_steps = 0
    current_action_idx = 0

    try:
        for episode in range(start_episode, 10000):
            ale.reset_game()
            state = preprocess_frame(ale.getScreenRGB())
            state = np.reshape(state, (1, 84, 84, 1))
            episode_reward = 0

            while not ale.game_over():

                if total_steps % 4 == 0:
                    if np.random.rand() <= epsilon:
                        current_action_idx = random.randrange(action_size)
                    else:
                        q_values = main_model(state, training=False)
                        current_action_idx = np.argmax(q_values[0])

                action = action_set[current_action_idx]
                #print(action)
                reward = ale.act(action)

                next_raw = ale.getScreenRGB()
                next_state = preprocess_frame(next_raw)
                next_state = np.reshape(next_state, (1, 84, 84, 1))

                memory.append((state, current_action_idx, reward, next_state, ale.game_over()))
                state = next_state

                episode_reward += reward
                total_steps += 1

                # TRAINING STEP
                if total_steps % 4 == 0 and len(memory) > BATCH_SIZE:
                    minibatch = random.sample(memory, BATCH_SIZE)

                    states_b = np.array([m[0][0] for m in minibatch])
                    next_states_b = np.array([m[3][0] for m in minibatch])

                    targets = main_model(states_b, training=False).numpy()
                    next_q = target_model(next_states_b, training=False).numpy()

                    for i, (_, a, r, _, d) in enumerate(minibatch):
                        targets[i][a] = r if d else r + GAMMA * np.max(next_q[i])

                    main_model.fit(states_b, targets, epochs=1, verbose=0)

                if total_steps % TARGET_UPDATE_FREQ == 0:
                    target_model.set_weights(main_model.get_weights())

                if total_steps % 20 == 0:
                    display_img = cv2.resize(next_raw, (400, 400))
                    cv2.imshow("Pong AI - TRAINING", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

            if epsilon > EPSILON_MIN:
                epsilon *= EPSILON_DECAY

            print(f"Ep: {episode} | Score: {episode_reward} | Epsilon: {epsilon:.4f}")

            if episode % 5 == 0:
                save_progress(main_model, epsilon, episode)

    except KeyboardInterrupt:
        print("\nTraining stopped.")
        save_progress(main_model, epsilon, episode)
    finally:
        cv2.destroyAllWindows()

# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run trained model in test mode")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        run_training()
