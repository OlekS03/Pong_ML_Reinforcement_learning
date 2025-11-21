import time
import random
import numpy as np
import cv2

from ale_py import ALEInterface, roms


def main(rom_name="pong"):
    ale = ALEInterface()

    # Recommended settings
    ale.setInt("random_seed", 123)
    ale.setBool("sound", False)
    ale.setFloat("repeat_action_probability", 0.0)

    # Load ROM
    rom_path = roms.get_rom_path(rom_name)
    ale.loadROM(rom_path)

    ale.reset_game()

    action_set = ale.getMinimalActionSet()
    print("Action set:", action_set)

    win_name = f"ALE: {rom_name}"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    episode = 0
    total_score = 0   # we track score manually

    try:
        while True:

            # Reset on episode end
            if ale.game_over():
                print(f"Episode {episode} finished. Score: {total_score}")
                episode += 1
                total_score = 0
                ale.reset_game()
                time.sleep(0.5)

            # Random agent
            action = random.choice(action_set)

            # ALE returns reward directly
            reward = ale.act(action)
            total_score += reward

            # Get screen
            screen = ale.getScreenRGB()
            frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            display = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

            # Show information
            text = (
                f"Episode: {episode}  "
                f"Frame: {frame_count}  "
                f"Reward: {reward:.1f}  "
                f"Score: {total_score}"
            )
            cv2.putText(display, text, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 1, cv2.LINE_AA)

            # Display window
            cv2.imshow(win_name, display)
            frame_count += 1

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit requested.")
                break

            time.sleep(0.01)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main("pong")
