import argparse
import module.memory as memory

from module import model_dirs, log_dirs, old_agents, new_agents


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--iftrain", type=bool, default=True, help="train or not")
    # parser.add_argument("--iftrain", type=bool, default=False, help="train or not")

    parser.add_argument("--scenario", type=str, default="simple_push_box", help="name of the scenario script")

    parser.add_argument("--len-episode", type=int, default=80, help="maximum episode length")
    parser.add_argument("--test-period", type=int, default=20, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=6000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--buffer-size", type=int, default=100000, help="length of replay buffer")
    parser.add_argument("--batch-size", type=int, default=64, help="size of mini batch")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lr-actor", type=float, default=0.001, help="learning rate for actor")
    parser.add_argument("--lr-critic", type=float, default=0.001, help="learning rate for critic")
    parser.add_argument("--explore-sigma", type=float, default=0.5, help="sigma: explore noise std")
    parser.add_argument("--tau", type=float, default=0.001, help="tau: soft update rate")
    parser.add_argument("--move-bound", type=float, default=8.0, help="agent move range")
    # Added for incremental number of agents
    parser.add_argument("--phase", type=int, default=0)
    return parser.parse_args()


def make_env(scenario_name, num_old=2, num_new=0, benchmark=False):
    from multiagent_local.environment import MultiAgentEnv
    import multiagent_local.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num_old=num_old, num_new=num_new)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,
                            scenario.done)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
    return env


def run(arglist):
    arglist.n_olds = old_agents[arglist.phase]
    arglist.n_newers = new_agents[arglist.phase]
    # Create environment
    env = make_env(arglist.scenario, num_old=arglist.n_olds, num_new=arglist.n_newers)
    arglist.n_agents = arglist.n_olds + arglist.n_newers
    arglist.dim_o = env.observation_space[0].shape[0]
    arglist.dim_a = 2
    arglist.actor_net_h_unit = [64, 64]
    arglist.critic_net_h_unit = [64, 64]
    arglist.n_old_agents = arglist.n_agents - arglist.n_newers
    arglist.obs_shape = (arglist.n_agents, arglist.dim_o)
    arglist.act_shape = (arglist.n_agents, arglist.dim_a)
    arglist.reward_shape = (arglist.n_agents, 1)
    arglist.terminal_shape = (arglist.n_agents, 1)

    if arglist.phase == 0:
        from module import train_pre as train, maddpg as ddpg_model
        arglist.dim_o_pre = arglist.dim_o
        arglist.dim_a_pre = arglist.dim_a
        arglist.value_shape = (arglist.n_agents, 1)
        arglist.a_adv_shape = (arglist.n_agents, arglist.dim_a)
        replay_buffer = memory.Memory(limit=arglist.buffer_size,
                                      observation_shape=arglist.obs_shape,
                                      action_shape=arglist.act_shape,
                                      reward_shape=arglist.reward_shape,
                                      terminal_shape=arglist.terminal_shape)
        Agents_pre = []
        for idx_agt in range(arglist.n_agents):
            Agents_pre.append(ddpg_model.Agent_model(agent_ID=idx_agt + 1,
                                                     args=arglist,
                                                     memory=replay_buffer))

        Trainer = train.Train_engine(Agents=Agents_pre,
                                     Memory=replay_buffer,
                                     args=arglist)
        Trainer.run(env=env,
                    len_episodes=arglist.len_episode,
                    num_episode=arglist.num_episodes,
                    test_period=arglist.test_period,
                    model_path=model_dirs[arglist.phase],
                    log_path=log_dirs[arglist.phase],
                    is_Train=arglist.iftrain)

    else:
        import module.maddpg as ddpg_model
        from module import train_icm as train, maddpg_icm_trans as ddpg_model_icm
        arglist.dim_o_pre = arglist.dim_o - 2 * arglist.n_newers
        arglist.dim_a_pre = arglist.dim_a
        arglist.n_old_agents = arglist.n_agents - arglist.n_newers
        arglist.pre_old = 2
        arglist.value_shape = (arglist.n_agents - arglist.pre_old, arglist.pre_old, 1)
        arglist.a_adv_shape = (arglist.n_agents - arglist.pre_old, arglist.pre_old, arglist.dim_a)

        arglist.value_old_shape = (arglist.n_old_agents, 1)
        arglist.act_old_shape = (arglist.n_old_agents, arglist.dim_a)

        replay_buffer = memory.Memory(limit=arglist.buffer_size,
                                      observation_shape=arglist.obs_shape,
                                      action_shape=arglist.act_shape,
                                      reward_shape=arglist.reward_shape,
                                      terminal_shape=arglist.terminal_shape,
                                      Q_value_shape=arglist.value_shape,
                                      act_adv_shape=arglist.a_adv_shape,
                                      Q_old_shape=arglist.value_old_shape,
                                      act_old_shape=arglist.act_old_shape)
        Agents_pre = []
        for idx_agt in range(arglist.n_old_agents):
            Agents_pre.append(ddpg_model.Agent_model(agent_ID=idx_agt + 1,
                                                     args=arglist,
                                                     memory=replay_buffer))
        Agents = []
        for idx_agt in range(arglist.n_agents):
            Agents.append(ddpg_model_icm.Agent_model(agent_ID=idx_agt + 1,
                                                     args=arglist,
                                                     memory=replay_buffer))

        Trainer = train.Train_engine(agts=Agents,
                                     agts_pre=Agents_pre,
                                     Memory=replay_buffer,
                                     args=arglist)
        Trainer.run(env=env,
                    len_episodes=arglist.len_episode,
                    num_episode=arglist.num_episodes,
                    test_period=arglist.test_period,
                    model_path=model_dirs[arglist.phase],
                    model_path_pre=model_dirs[arglist.phase-1],
                    log_path=log_dirs[arglist.phase],
                    is_Train=arglist.iftrain)


if __name__ == '__main__':
    arglist = parse_args()
    run(arglist)
