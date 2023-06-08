import numpy as np
import tensorflow as tf
import time
from copy import deepcopy
import pandas as pd


class Train_engine():
    def __init__(self, agts, agts_pre, Memory, args):
        self.agents = agts
        self.agents_pre = agts_pre
        self.replay_buffer = Memory
        self.batch_size = args.batch_size
        self.n_agents = args.n_agents
        self.n_others = self.n_agents - 1
        self.n_newers = args.n_newers
        self.n_old = self.n_agents - self.n_newers
        self.d_o = self.agents[0].dim_o
        self.d_a = self.agents[0].dim_a
        self.move_range = args.move_bound
        self.label_pre = "Phase_" + str(args.phase-1) + "_"
        self.label_tar = "Phase_" + str(args.phase) + "_"
        self.n_old_pre = args.pre_old

    def run(self, env, len_episodes, num_episode, test_period, model_path, model_path_pre, log_path, is_Train=False):
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
                    self.Total_reward_sum.append(
                        tf.summary.scalar("Total_reward_agt_" + str(idx_agt + 1), self.Total_reward[idx_agt]))
                merge_reward = tf.summary.merge(self.Total_reward_sum)
                writer = tf.summary.FileWriter(log_path, sess.graph)

            # restore current model
            self.saver = tf.train.Saver()
            model_step = 0
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                model_step = int(ckpt.model_checkpoint_path[-1])

            # restore previous model
            ckpt_pre = tf.train.get_checkpoint_state(model_path_pre)
            # get historical variables to restore
            variables_all = tf.contrib.framework.get_variables_to_restore()
            variables_pre = [v for v in variables_all if self.label_pre in v.name]
            self.saver_pre = tf.train.Saver(variables_pre)
            if ckpt_pre and ckpt_pre.model_checkpoint_path:
                self.saver_pre.restore(sess, ckpt_pre.model_checkpoint_path)

            # load the trained parameters from previous environments to current environment
            for idx_agt in range(self.n_old):
                # actor network
                for idx_par, param_a in enumerate(self.agents[idx_agt].param_a):
                    if idx_par == 0:  # input layer
                        sess.run(param_a.assign(np.zeros(shape=param_a.shape)))
                        # sess.run(param_a.assign(np.random.normal(loc=0.0, scale=0.3, size=param_a.shape)))
                        index_new = [0, self.agents_pre[idx_agt].dim_o]
                        index_old = deepcopy(index_new)
                        par_assign = tf.assign(param_a[index_new[0]:index_new[1], :],
                                               self.agents_pre[idx_agt].param_a[idx_par][index_old[0]:index_old[1], :])
                    else:
                        par_assign = tf.assign(param_a, self.agents_pre[idx_agt].param_a[idx_par])
                    sess.run(par_assign)
                # critic network
                for idx_par, param_c in enumerate(self.agents[idx_agt].param_c):
                    if idx_par == 0:  # input layer
                        sess.run(param_c.assign(np.zeros(shape=param_c.shape)))
                        # sess.run(param_c.assign(np.random.normal(loc=0.0, scale=0.3, size=param_c.shape)))
                        # self obs
                        index_new = [0, self.agents_pre[idx_agt].dim_o]
                        index_old = deepcopy(index_new)
                        par_assign_1 = tf.assign(param_c[index_new[0]:index_new[1], :],
                                                 self.agents_pre[idx_agt].param_c[idx_par][index_old[0]:index_old[1],
                                                 :])
                        sess.run(par_assign_1)
                        # others obs
                        for idx_others in range(self.n_others - self.n_newers):
                            index_new = [
                                self.agents[idx_agt].dim_o + self.agents[idx_agt].dim_a + idx_others * self.agents[
                                    idx_agt].dim_o,
                                self.agents[idx_agt].dim_o + self.agents[idx_agt].dim_a + idx_others * self.agents[
                                    idx_agt].dim_o + self.agents_pre[idx_agt].dim_o]
                            index_old = [
                                self.agents_pre[idx_agt].dim_o + self.agents_pre[idx_agt].dim_a + idx_others *
                                self.agents_pre[
                                    idx_agt].dim_o,
                                self.agents_pre[idx_agt].dim_o + self.agents_pre[idx_agt].dim_a + idx_others *
                                self.agents_pre[
                                    idx_agt].dim_o + self.agents_pre[idx_agt].dim_o]
                            par_assign_2 = tf.assign(param_c[index_new[0]:index_new[1], :],
                                                     self.agents_pre[idx_agt].param_c[idx_par][
                                                     index_old[0]:index_old[1],
                                                     :])
                            sess.run(par_assign_2)
                        # others act
                        for idx_others in range(self.n_others - self.n_newers):
                            index_new = [
                                self.agents[idx_agt].dim_o + self.agents[idx_agt].dim_a + self.n_others * self.agents[
                                    idx_agt].dim_o + idx_others * self.agents[idx_agt].dim_a,
                                self.agents[idx_agt].dim_o + self.agents[idx_agt].dim_a + self.n_others * self.agents[
                                    idx_agt].dim_o + idx_others * self.agents[idx_agt].dim_a + self.agents_pre[
                                    idx_agt].dim_a
                            ]
                            index_old = [
                                self.agents_pre[idx_agt].dim_o + self.agents_pre[idx_agt].dim_a + (self.n_old - 1) *
                                self.agents_pre[
                                    idx_agt].dim_o + idx_others * self.agents_pre[idx_agt].dim_a,
                                self.agents_pre[idx_agt].dim_o + self.agents_pre[idx_agt].dim_a + (self.n_old - 1) *
                                self.agents_pre[
                                    idx_agt].dim_o + idx_others * self.agents_pre[idx_agt].dim_a + self.agents_pre[
                                    idx_agt].dim_a
                            ]
                            par_assign_3 = tf.assign(param_c[index_new[0]:index_new[1], :],
                                                     self.agents_pre[idx_agt].param_c[idx_par][
                                                     index_old[0]:index_old[1],
                                                     :])
                            sess.run(par_assign_3)

                    else:
                        par_assign = tf.assign(param_c, self.agents_pre[idx_agt].param_c[idx_par])
                        sess.run(par_assign)
                self.agents[idx_agt].update_target_net(sess=sess, init=True)

            act_t = np.zeros([self.n_agents, self.agents[0].dim_a])
            act_next = np.zeros([self.batch_size, self.n_agents, self.d_a])
            act_t_adv = np.zeros([self.n_agents-self.n_old_pre, self.n_old_pre, self.agents[0].dim_a])
            q_advice = np.zeros([self.n_agents-self.n_old_pre, self.n_old_pre, 1])
            act_t_adv_old = np.zeros([self.n_old, self.agents[0].dim_a])
            q_advice_old = np.zeros([self.n_old, 1])
            loss_a = np.zeros([self.n_agents])
            loss_c = np.zeros([self.n_agents])

            history_data = np.zeros([int(num_episode / test_period), self.n_agents])
            print("Start training")
            for idx_epo in range(num_episode):
                obs_t = env.reset()

                ############  test  ############
                if not is_Train:
                    # env.dampling = 0.9999
                    for idx_step in range(len_episodes):
                        # get actions for each agent
                        for idx_agt in range(self.n_agents):
                            act_t[idx_agt] = self.agents[idx_agt].get_action(observation=[obs_t[idx_agt]],
                                                                             sess=sess,
                                                                             noise=False)
                        act_step_t = self.joint_action(act_t)
                        obs_next, reward, done, info = env.step(act_step_t)
                        obs_t = deepcopy(obs_next)
                        env.render()
                        time.sleep(0.1)
                    continue

                ############  train  ############
                for idx_step in range(len_episodes):
                    # get actions for each agent
                    for idx_agt in range(self.n_agents):
                        act_t[idx_agt] = self.agents[idx_agt].get_action(observation=[obs_t[idx_agt]],
                                                                         sess=sess,
                                                                         noise=True)
                    # get adviced actions and Q_values
                    for idx_agt in range(self.n_old_pre):  # student in previous never be teacher
                        # for i in range(self.n_newers):
                        for i in range(self.n_agents-self.n_old_pre):
                            idx_new = i + self.n_old_pre  # i + self.n_old
                            obs_others = np.delete(obs_t, idx_new, axis=0).reshape([1, -1])
                            act_others = np.delete(act_t, idx_new, axis=0).reshape([1, -1])
                            act_t_adv[i, idx_agt] = self.agents[idx_agt].get_action(
                                observation=[obs_t[idx_new]],
                                sess=sess,
                                noise=False)
                            q_advice[i, idx_agt] = self.agents[idx_agt].get_q_values(obs=[obs_t[idx_new]],
                                                                                     obs_others=obs_others,
                                                                                     act=[act_t[idx_new]],
                                                                                     act_others=act_others,
                                                                                     sess=sess)

                    act_step_t = self.joint_action(act_t)
                    obs_next, reward, done, info = env.step(act_step_t)
                    # env.render()
                    self.replay_buffer.append(obs0=obs_t, action=act_t, reward=reward, obs1=obs_next, terminal1=done,
                                              q_adv=q_advice, a_adv=act_t_adv,
                                              q_old=q_advice_old, a_old=act_t_adv_old,
                                              training=is_Train)
                    obs_t = deepcopy(obs_next)

                    if self.replay_buffer.nb_entries < self.batch_size:
                        continue

                    samples = self.replay_buffer.sample(batch_size=self.batch_size)
                    a_adv_sample = samples['a_advice']
                    q_adv_sample = samples['q_advice']

                    a_adv_old_sample = samples['a_old']
                    q_adv_old_sample = samples['q_old']

                    for idx_agt in range(self.n_agents):
                        act_next[:, idx_agt, :] = self.agents[idx_agt].act_next.eval(
                            feed_dict={self.agents[idx_agt].obs_next: samples['obs1'][:, idx_agt, :]}, session=sess)
                    # update critic network
                    for idx_agt, agent in enumerate(self.agents):
                        o_others = np.delete(samples['obs0'], idx_agt, axis=1).reshape([self.batch_size, -1])
                        a_others = np.delete(samples['actions'], idx_agt, axis=1).reshape([self.batch_size, -1])
                        o_next_others = np.delete(samples['obs1'], idx_agt, axis=1)
                        a_next_others = np.delete(act_next, idx_agt, axis=1)
                        q_predict = agent.get_q_predict(r=samples['rewards'][:, idx_agt, :],
                                                        obs_next=samples['obs1'][:, idx_agt, :],
                                                        obs_next_others=o_next_others.reshape(self.batch_size, -1),
                                                        act_next_others=a_next_others.reshape(self.batch_size, -1),
                                                        sess=sess)
                        # get sampled advice actions and values for New Agent
                        if idx_agt < self.n_old_pre:
                            Q_teacher = q_adv_old_sample[:, idx_agt]
                            A_teacher = a_adv_old_sample[:, idx_agt, :]
                        else:
                            idx = idx_agt - self.n_old_pre
                            Q_teacher = np.max(q_adv_sample[:, idx, :, :], axis=1)
                            a_adv_idx = np.argmax(q_adv_sample[:, idx, :, :], axis=1)
                            A_teacher = np.reshape(
                                [a_adv_sample[i, idx, a_adv_idx[i], :] for i in range(self.batch_size)],
                                [self.batch_size, agent.dim_a])

                        _, loss_c[idx_agt] = sess.run([agent.trainer_c, agent.loss_c],
                                                      feed_dict={agent.obs_t: samples['obs0'][:, idx_agt, :],
                                                                 agent.act_t: samples['actions'][:, idx_agt, :],
                                                                 agent.obs_others: o_others,
                                                                 agent.act_others: a_others,
                                                                 agent.Q_predict: q_predict,
                                                                 agent.Q_teacher: Q_teacher})
                        # update actor network
                        _, loss_a[idx_agt] = sess.run([agent.trainer_a, agent.loss_a],
                                                      feed_dict={agent.obs_t: samples['obs0'][:, idx_agt, :],
                                                                 agent.obs_others: o_others,
                                                                 agent.act_others: a_others,
                                                                 agent.act_teacher: A_teacher})
                        agent.update_target_net(sess=sess, init=False)

                        # print("step, loss_c, loss_a: ", idx_step*len_episodes+idx_step, loss_c, loss_a)
                    # if done:
                    #     obs_t = env.reset()

                #####################################################
                # # # # # # # # #  Get Performance  # # # # # # # # #
                #####################################################

                if idx_epo % test_period == 0:
                    total_reward = np.zeros([self.n_agents])
                    num_test = 10
                    # env.world.collide_penalty = False
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

                    # env.world.collide_penalty = True
                    ave_total_reward = np.divide(total_reward, num_test)
                    summary = sess.run(merge_reward, feed_dict={self.Total_reward: ave_total_reward})
                    writer.add_summary(summary, idx_epo)

                    # save as .csv directly
                    for idx_agt in range(self.n_agents):
                        history_data[model_step] = deepcopy(ave_total_reward)

                    end = time.time()
                    print('Epo: %5d, mean_episode_reward: %.4f, Time: %.2fs' % (idx_epo, np.mean(ave_total_reward), end-start))

                    # save model
                    model_step += 1
                    self.saver.save(sess, model_path, global_step=model_step)

            for idx_agt in range(self.n_agents):
                saved_to_csv = pd.DataFrame(columns=["Value"], data=history_data[:, idx_agt])
                saved_to_csv.to_csv(model_path + "/run_.-tag-Total_reward_agt_" + str(idx_agt + 1) + ".csv")

    def joint_action(self, acts):
        act_joint = np.zeros([self.n_agents, 5])
        act_joint[:, [1, 3]] = acts
        return act_joint

    def is_out_of_range(self, state):
        pos = [state[0][2], state[0][3], state[1][2], state[1][3], state[2][2], state[2][3], state[3][2], state[3][3]]
        if np.max(pos) > self.move_range:
            return True
        else:
            return False
