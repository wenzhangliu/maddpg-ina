import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd

class Train_engine():
    def __init__(self, Agents, Memory, args):
        self.agents = Agents
        self.replay_buffer = Memory
        self.batch_size = args.batch_size
        self.n_agents = len(Agents)
        self.n_others = self.n_agents - 1
        self.n_newers = 1
        self.d_o = self.agents[0].dim_o
        self.d_a = self.agents[0].dim_a
        self.move_range = args.move_bound

    def run(self, env, len_episodes, num_episode, test_period, model_path, log_path, is_Train=False):
        start = time.time()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for idx_agt in range(self.n_agents):
                self.agents[idx_agt].update_target_net(sess=sess, init=True)

            if is_Train:
                self.Total_reward = tf.placeholder(tf.float32, [self.n_agents], "total_reward")
                self.Total_reward_sum = []
                for idx_agt in range(self.n_agents):
                    self.Total_reward_sum.append(tf.summary.scalar("Total_reward_agt_"+str(idx_agt+1), self.Total_reward[idx_agt]))
                merge_reward = tf.summary.merge([self.Total_reward_sum])
                writer = tf.summary.FileWriter(log_path, sess.graph)

            self.saver = tf.train.Saver()
            model_step = 0
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                model_step = int(ckpt.model_checkpoint_path[-1])

            act_t = np.zeros([self.n_agents, self.agents[0].dim_a])
            act_next = np.zeros([self.batch_size, self.n_agents, self.d_a])
            loss_a = np.zeros([self.n_agents])
            loss_c = np.zeros([self.n_agents])

            history_data = np.zeros([int(num_episode / test_period), self.n_agents])
            print("Start training")
            for idx_epo in range(num_episode):
                obs_t = env.reset()

                if not is_Train:
                    # env.dampling = 0.9999
                    for idx_step in range(len_episodes):
                        # get actions for each agent
                        for idx_agt in range(self.n_agents):
                            act_t[idx_agt] = self.agents[idx_agt].get_action(observation=[obs_t[idx_agt]],
                                                                             sess=sess,
                                                                             noise=False)
                        # act_t[1] = np.array([0.0, 0.0])
                        print(act_t[0], act_t[1])
                        # print(env.agents[0].state.p_vel, env.agents[1].state.p_vel)
                        act_step_t = self.joint_action(act_t)
                        obs_next, reward, done, info = env.step(act_step_t)
                        obs_t = deepcopy(obs_next)
                        env.render()
                        time.sleep(0.02)
                    continue

                for idx_step in range(len_episodes):
                    # get actions for each agent
                    for idx_agt in range(self.n_agents):
                        act_t[idx_agt] = self.agents[idx_agt].get_action(observation=[obs_t[idx_agt]],
                                                                         sess=sess,
                                                                         noise=True)
                    act_step_t = self.joint_action(act_t)
                    obs_next, reward, done, info = env.step(act_step_t)
                    self.replay_buffer.append(obs0=obs_t, action=act_t, reward=reward, obs1=obs_next, terminal1=done, training=is_Train)
                    obs_t = deepcopy(obs_next)
                    # env.render()

                    # print(env.agents[0].state.p_vel, env.agents[1].state.p_vel)

                    # if (True in done): break

                    if self.replay_buffer.nb_entries < self.batch_size:
                        continue

                    samples = self.replay_buffer.sample(batch_size=self.batch_size)

                    for idx_agt in range(self.n_agents):
                        act_next[:, idx_agt, :] = self.agents[idx_agt].act_next.eval(
                            feed_dict={self.agents[idx_agt].obs_next: samples['obs1'][:, idx_agt, :]})

                    for idx_agt, agent in enumerate(self.agents):
                        o_others = np.delete(samples['obs0'], idx_agt, axis=1)
                        a_others = np.delete(samples['actions'], idx_agt, axis=1)
                        o_next_others = np.delete(samples['obs1'], idx_agt, axis=1)
                        a_next_others = np.delete(act_next, idx_agt, axis=1)
                        q_predict = agent.get_q_predict(r=samples['rewards'][:, idx_agt, :],
                                                        obs_next=samples['obs1'][:, idx_agt, :],
                                                        obs_next_others=o_next_others.reshape(self.batch_size, -1),
                                                        act_next_others=a_next_others.reshape(self.batch_size, -1),
                                                        sess=sess)
                        # update critic network
                        _, loss_c[idx_agt] = sess.run([agent.trainer_c, agent.loss_c],
                                                      feed_dict={agent.obs_t: samples['obs0'][:, idx_agt, :],
                                                                 agent.act_t: samples['actions'][:, idx_agt, :],
                                                                 agent.obs_others: o_others.reshape(self.batch_size, -1),
                                                                 agent.act_others: a_others.reshape(self.batch_size, -1),
                                                                 agent.Q_predict: q_predict
                                                                 })
                        # update actor network
                        _, loss_a[idx_agt] = sess.run([agent.trainer_a, agent.loss_a],
                                                      feed_dict={agent.obs_t: samples['obs0'][:, idx_agt, :],
                                                                 agent.obs_others: o_others.reshape(self.batch_size, -1),
                                                                 agent.act_others: a_others.reshape(self.batch_size, -1)
                                                                 })

                        agent.update_target_net(sess=sess, init=False)

                        # print("step, loss_c, loss_a: ", idx_step*len_episodes+idx_step, loss_c, loss_a)

                if idx_epo % test_period == 0:
                    env.world.base_reward = True
                    total_reward = np.zeros(self.n_agents)
                    num_test = 10
                    for idx_test in range(num_test):
                        obs_t = env.reset()
                        for t in range(len_episodes):
                            for idx_agt in range(self.n_agents):
                                act_t[idx_agt] = self.agents[idx_agt].get_action(observation=[obs_t[idx_agt]],
                                                                                 sess=sess,
                                                                                 noise=False)
                            act_step_t = self.joint_action(act_t)
                            obs_next, reward, done, info = env.step(act_step_t)
                            total_reward = total_reward + np.reshape(reward, [self.n_agents])
                            obs_t = deepcopy(obs_next)

                    env.world.base_reward = False
                    ave_total_reward = np.divide(total_reward, num_test)
                    summary = sess.run(merge_reward, feed_dict={self.Total_reward: ave_total_reward})
                    writer.add_summary(summary, idx_epo)

                    # save as .csv directly
                    for idx_agt in range(self.n_agents):
                        history_data[model_step] = deepcopy(ave_total_reward)

                    end = time.time()
                    print('Epo: %5d, mean_episode_reward: %.4f, time: %.2fs'%(idx_epo, np.mean(ave_total_reward), end-start))

                    # save model
                    model_step += 1
                    self.saver.save(sess, model_path, global_step=model_step)

            for idx_agt in range(self.n_agents):
                saved_to_csv = pd.DataFrame(columns=["Value"], data=history_data[:, idx_agt])
                saved_to_csv.to_csv(model_path + "run_.-tag-Total_reward_agt_" + str(idx_agt + 1) + ".csv")

    def joint_action(self, acts):
        act_joint = np.zeros([self.n_agents, 5])
        act_joint[:, [1, 3]] = acts
        return act_joint

    def is_out_of_range(self):
        pos = [agent.state.p_pos for agent in self.agents]
        if np.max(pos) > self.move_range:
            return True
        else:
            return False

