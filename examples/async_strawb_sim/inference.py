import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
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
    env = gym.make("gym_INB0104/ReachIKDeltaStrawbHangingEnv", width=width, height=height, cameras=["wrist1", "wrist2"], randomize_domain=True, ee_dof=6)
    env = TimeLimit(env, max_episode_steps=50)
    env = SERLObsWrapper(env)
    env = ActionState(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    waitkey = 10
    resize_resolution = (720, 720)

    # Define the path to your checkpoint
    checkpoint_path = "/home/emlyn/rl_franka/serl/examples/async_strawb_sim/checkpoints_3/checkpoint_40000_20241202_182431"

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
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            if render_mode == "rgb_array":
                wrist1 = obs["wrist1"][0]
                wrist1 = cv2.resize(wrist1, resize_resolution)
                wrist2 = obs["wrist2"][0]
                wrist2 = cv2.resize(wrist2, resize_resolution)
                cv2.imshow("pixels", cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR))
                cv2.imshow("pixels2", cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR))
                cv2.waitKey(waitkey)

            # Sample action from the restored agent
            rng, key = jax.random.split(rng)
            action = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                deterministic=True,  # Evaluation mode should be deterministic
            )
            action = np.asarray(jax.device_get(action))
            print(f"action: {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            i+=1
        
if __name__ == "__main__":
    main()
