import argparse
import numpy as np
import tensorflow as tf
import pickle
import memory
import train_pre

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

Task_decompose = True
# Task_decompose = False
Train = True
# Train = False

import ddpg_mt as ddpg


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_intercept", help="name of the scenario script")
    parser.add_argument("--max-trail-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=3000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    # parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--exp-name", type=str, default="experiments", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    # parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    # Added for ddpg-single
    if Task_decompose:
        parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint_td/")
    else:
        parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint/")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def run(arglist):
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    action_shape = (2,)
    obs_shape = (8,)

    if Task_decompose:
        reward_shape = (2,)
    else:
        reward_shape = (1,)

    buffer_size = 102400
    replay_buffer_att = memory.Memory(limit=buffer_size, observation_shape=obs_shape, action_shape=action_shape,
                                      reward_shape=reward_shape)
    replay_buffer_ict = memory.Memory(limit=buffer_size, observation_shape=obs_shape, action_shape=action_shape,
                                      reward_shape=reward_shape)

    if Task_decompose:
        attacker = ddpg.Agent_model(memory=replay_buffer_att, scope="Attacker", isTD=True)
        intercept = ddpg.Agent_model(memory=replay_buffer_ict, scope="Intercept", isTD=True)
        Train_engine = train_pre.train_engine_td(agent1=attacker, agent2=intercept, train=Train)
    else:
        attacker = ddpg.Agent_model(memory=replay_buffer_att, scope="Attacker", isTD=False)
        intercept = ddpg.Agent_model(memory=replay_buffer_ict, scope="Intercept", isTD=False)
        Train_engine = train_pre.train_engine(agent1=attacker, agent2=intercept, train=Train)

    Train_engine.run_them(env, episode=arglist.num_episodes,
                          Time_scale=arglist.max_trail_len, ckpt_path=arglist.checkpoint_dir)


if __name__ == '__main__':
    arglist = parse_args()
    run(arglist)
