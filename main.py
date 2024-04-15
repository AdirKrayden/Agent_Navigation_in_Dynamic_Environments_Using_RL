# Created by Adir Krayden on 02-03-24
# -------------------------------- Imports -------------------------------- #
from Environment import Environment
# from DQN import DQN
from gym.spaces import space
from time import sleep
import argparse
import os
import multiprocessing
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
import torch

# -------------------------------- Runs Examples -------------------------------- #

""""
Run (with parameters) examples (*MORE FLAGS ARE OPTIONAL*):

Train:
    --num_episodes 15000 --algorithm "PPO" --num_agents 1 --env_size 15
    --num_episodes 2500000 --algorithm "PPO" --num_agents 1 --env_size 10 --partial_observe
    --num_episodes 2500000 --algorithm "PPO" --num_agents 1 --env_size 10 --FOV_size 1
    --num_episodes 2500000 --algorithm "DQN" --num_agents 1 --env_size 10 --FOV_size 2 --partial_observe --reward_setup "sparse"

Inference/Test:
    --load_model  --model_path "models/Action space - all directions/sprase prizes/baseline3_DQN_agent_1_num_episodes_2500000_env_size_10_FOV_2.zip"  --env_size 15 --algorithm "DQN" --reward_setup "sparse" 
    --load_model --model_path "baseline3_PPO_agent_1_num_episodes_250000_env_size_10_FOV_2.zip" --FOV_size 2  --env_size 10 --partial_action --reward_setup "sparse"
    --load_model --model_path "models/Action space - all directions/sprase prizes/baseline3_PPO_agent_1_num_episodes_2500000_env_size_10_FOV_2_partial_obs.zip" --env_size 18  --reward_setup "sparse" --partial_observe 
    --load_model --model_path "baseline3_DQN_agent_1_num_episodes_50000_env_size_10_FOV_3.zip" --env_size 10 --algorithm "DQN"  --reward_setup "sparse"  --partial_action  --FOV_size 3 
"""


# -------------------------------- parser -------------------------------- #

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train or load a reinforcement learning model for a custom"
                                                 " environment.")
    parser.add_argument("--load_model", action="store_true", default=False,
                        help="Whether to load a pre-trained model or not")
    parser.add_argument("--model_path", type=str, default=None, help="Path to load/save the model")
    parser.add_argument("--num_episodes", type=int, default=50000,
                        help="Number of training episodes - an integer in range [50000, 5000000]")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=['DQN', 'PPO'],
                        help="Model algorithm to train on - DQN or PPO")
    parser.add_argument("--reward_setup", type=str, default="reg", choices=['reg', 'sparse'],
                        help="Reward setup for the agent - either 'reg' or 'sparse' (gets positive reward for only reaching the prize)")
    parser.add_argument("--num_agents", type=int, default=1, choices=range(1, 4), help="Number of agents,"
                                                                                       " an integer in range [1, 3]")
    parser.add_argument("--FOV_size", type=int, default=2, choices=range(1, 4), help="2D FOV size -"
                                                                                     " a number in range [1, 3]")
    parser.add_argument("--env_size", type=int, default=10, choices=range(10, 51), help="Environment size,"
                                                                                        " an integer in range [10, 50")
    parser.add_argument("--partial_observe", action="store_true", default=False,
                        help="If true, the agent's observation space is only it's surroundings")
    parser.add_argument("--partial_action", action="store_true", default=False,
                        help="If true, the agent's action space is limited - moving up/turning to the right")
    return parser.parse_args()


# -------------------------------- main -------------------------------- #

def test_multiple_environments(args):
    env_num = 10000
    results = {}  # Dictionary to store results for each environment

    # List of model paths to test
    model_paths = [
        # "models/Action space - all directions/'regular' prizes/baseline3_PPO_agent_1_num_episodes_2500000_env_size_10_FOV_1.zip",
        # "models/Action space - all directions/'regular' prizes/baseline3_PPO_agent_1_num_episodes_2500000_env_size_10_FOV_2.zip",
        "models/Action space - all directions/sprase prizes/baseline3_DQN_agent_1_num_episodes_2500000_env_size_10_FOV_2_partial_obs.zip",
    ]

    # Load models
    models = {}
    for model_path in model_paths:
        if args.algorithm == 'DQN':
            models[model_path] = DQN.load(model_path)
        else:
            models[model_path] = PPO.load(model_path)

    for i in range(env_num):  # Generate environments
        if i % 100 == 0:
            print(i)
        env = Environment(matrix_shape=(args.env_size, args.env_size), partial_obs=args.partial_observe,
                          num_agents=args.num_agents, obs_interval=args.FOV_size, reward_setup=args.reward_setup,
                          partial_act=args.partial_action)

        obs, _ = env.reset()
        max_steps_per_episode = 5 * args.env_size

        for model_path, model in models.items():
            for step in range(max_steps_per_episode):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if done:
                    # Store the number of steps taken to reach the goal for this environment and model
                    if model_path not in results:
                        results[model_path] = {}
                    results[model_path][i] = step + 1
                    break

                elif (not done) and (step == max_steps_per_episode-1):
                    if model_path not in results:
                        results[model_path] = {}
                    results[model_path][i] = step + 1
                    break

    return results


def main(args):
    # Check if model_path is provided
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.load_model:
        if args.model_path is None:
            raise ValueError("You must provide a model path if you want to load a pre-trained model.")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"The specified model file '{args.model_path}' does not exist.")

    if not 50000 <= args.num_episodes <= 5000000:
        raise ValueError("Number of episodes must be between 50,000 and 5,000,000.")

    if args.algorithm not in ['DQN', 'PPO']:
        raise ValueError("Invalid model algorithm. Choose either 'DQN' or 'PPO'.")

    if args.reward_setup not in ['reg', 'sparse']:
        raise ValueError("Invalid prize setup. Choose either 'reg' or 'sparse'.")

    if not isinstance(args.num_agents, int):
        raise ValueError("agent num must be an integer.")
    if not 1 <= args.num_agents <= 3:
        raise ValueError("Number of agents in range [1,3].")

    if not isinstance(args.FOV_size, int):
        raise ValueError("Fov size must be an integer.")
    if not 1 <= args.FOV_size <= 3:
        raise ValueError("FOV size in range [1,3].")

    if not isinstance(args.env_size, int):
        raise ValueError("Environment size parameter must be an integer.")
    if not 10 <= args.env_size <= 50:
        raise ValueError("Environment size in range [10, 50].")

    print("Creating an environment..")
    print("The following environment variables are set:")
    print("Agent's FOV: {}".format(args.FOV_size))
    print("Environment size: {}x{}".format(args.env_size, args.env_size))
    print("partial_observe: {}".format(args.partial_observe))
    print("partial_action: {}".format(args.partial_action))
    print("num_agents: {}".format(args.num_agents))
    print("reward_setup: {}".format(args.reward_setup))
    if not args.load_model:
        print("Number of training episodes: {}".format(args.num_episodes))
    env = Environment(matrix_shape=(args.env_size, args.env_size), partial_obs=args.partial_observe,
                      num_agents=args.num_agents, obs_interval=args.FOV_size, reward_setup=args.reward_setup,
                      partial_act=args.partial_action)
    print("Environment is set!")

    if not args.load_model:  # train a new model
        if args.algorithm == 'DQN':
            model = DQN("MultiInputPolicy", env, device="cuda").learn(total_timesteps=args.num_episodes)
        else:
            model = PPO(policy="MultiInputPolicy",env=env, device=device, ent_coef=0.001).learn(total_timesteps=args.num_episodes)
        print("Finished training!")
        print("Saving the model...")
        if not args.partial_observe:
            model_name = "baseline3_" + str(args.algorithm) + "_agent_" + str(args.num_agents) + "_num_episodes_" + str(
                args.num_episodes) + "_env_size_" + str(args.env_size) + "_FOV_" + str(args.FOV_size)
        else:
            model_name = "baseline3_" + str(args.algorithm) + "_agent_" + str(args.num_agents) + "_num_episodes_" + str(
                args.num_episodes) + "_env_size_" + str(args.env_size) + "_FOV_" + str(args.FOV_size) + "_partial_obs"
        model.save(model_name)
        print("model saved as: " + str(model_name))
    else:  # testing/using a model
        if args.algorithm == 'DQN':
            model = DQN.load(args.model_path)
        else:
            model = PPO.load(args.model_path)
        obs, _ = env.reset()
        max_steps_per_episode = 7 * args.env_size
        for step in range(max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            print(f"Step {step + 1}")
            print("Action: ", action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            sleep(2)  # solve HTTP Error 429: Too Many Requests error
            print("obs=", obs, "reward=", reward, "done=", done)
            env.render()
            if done:
                print("Goal reached!", "reward=", reward)
                break


if __name__ == "__main__":
    args = parse_arguments()
    main(args)  # train / test environment

    # # Test multiple environments
    # results = test_multiple_environments(args)
    #
    # # Save the results to a file
    # with open("results.txt", "w") as f:
    #     for model_path, env_results in results.items():
    #         f.write(f"Model: {model_path}\n")
    #         for env_id, steps in env_results.items():
    #             f.write(f"  Environment {env_id}: Steps taken = {steps}\n")
    #
    # # Calculate average steps for each model
    # for model_path, env_results in results.items():
    #     average_steps = sum(env_results.values()) / len(env_results)
    #     print(f"Model {model_path}: Average steps across all environments = {average_steps}")

