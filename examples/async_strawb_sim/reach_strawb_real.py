import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
import numpy as np
from datetime import datetime
from franka_ros2_gym import envs

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.launcher import make_drq_agent

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.wrappers import ActionState

def main():
    render_mode = "rgb_array"
    height, width = 112, 112 
    env = gym.make("franka_ros2_gym/ReachIKDeltaRealStrawbEnv", width=width, height=height, render_mode=render_mode, pos_scale=0.05, cameras=['wrist1', 'wrist2'])
    env = TimeLimit(env, max_episode_steps=100)
    env = SERLObsWrapper(env)
    env = ActionState(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    waitkey = 10
    resize_resolution = (720, 720)

     # Define the path to your checkpoint
    checkpoint_path = "/home/emlyn/rl_franka/serl/examples/async_strawb_sim/checkpoints_5/checkpoint_800000_20241207_095713"

    # Create agent instance
    seed = 42
    rng = jax.random.PRNGKey(seed)
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    agent: DrQAgent = make_drq_agent(
        seed=seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type="resnet-pretrained",
    )

    # Restore the latest checkpoint
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=agent.state)
    agent = agent.replace(state=restored_state)

    # Replicate agent across devices if needed
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    print("Checkpoint restored successfully!")

    while True:
        terminated = False
        truncated = False
        obs, info = env.reset()
        
        while not terminated and not truncated:
            # Display the environment
            if render_mode == "rgb_array":

                # Show the wrist camera view
                wrist1_pixels = obs["wrist1"][0]
                wrist2_pixels = obs["wrist2"][0]
                cv2.imshow("wrist_camera", cv2.resize(cv2.cvtColor(wrist1_pixels, cv2.COLOR_RGB2BGR), resize_resolution))
                cv2.imshow("wrist_camera2", cv2.resize(cv2.cvtColor(wrist2_pixels, cv2.COLOR_RGB2BGR), resize_resolution))
                cv2.waitKey(waitkey)
            
           
            # Sample action from the restored agent
            rng, key = jax.random.split(rng)
            action = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                deterministic=True,  # Evaluation mode should be deterministic
            )
            action = np.asarray(jax.device_get(action)).copy()
            action[2] += 0.02
            print(action)
            # Perform the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)



if __name__ == "__main__":
    main()
