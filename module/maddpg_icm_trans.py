import numpy as np
import tensorflow as tf


class Agent_model():
    def __init__(self, agent_ID, args, memory):
        self.ID = agent_ID
        self.dim_o = args.dim_o
        self.dim_a = args.dim_a
        self.n_agents = args.n_agents
        self.n_others = self.n_agents - 1
        self.n_newers = args.n_newers
        self.n_old = self.n_agents - self.n_newers
        self.n_old_pre = args.pre_old
        # self.n_newers = args.n_newers
        self.dim_a_others = self.n_others * self.dim_a
        self.gamma = args.gamma
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.tau = args.tau
        self.memory = memory
        self.memory_size = memory.limit
        self.dim_units_a = args.actor_net_h_unit
        self.dim_units_c = args.critic_net_h_unit

        # exploration setting
        # exploration noise
        self.mu = []
        self.sigma = args.explore_sigma * np.eye(self.dim_a, self.dim_a)

        self.weight_distill = 20.0

        self.label = "Phase_" + str(args.phase) + "_"

        # Input and Output For Actor Network
        self.obs_t = tf.placeholder(tf.float32, [None, self.dim_o], name=self.label+"obs_t_agt_" + str(self.ID))
        self.obs_next = tf.placeholder(tf.float32, [None, self.dim_o], name=self.label+"obs_next_agt_" + str(self.ID))
        self.act_t = tf.placeholder(tf.float32, [None, self.dim_a], name=self.label+"act_t_agt_" + str(self.ID))
        self.act_t = self.actor_net(obs_in=self.obs_t, scope=self.label+"actor_net_agt_" + str(self.ID))
        self.act_next = self.actor_net(obs_in=self.obs_next, scope=self.label+"actor_target_net_agt_" + str(self.ID))
        self.obs_others = tf.placeholder(tf.float32, [None, self.dim_o * self.n_others],
                                         name=self.label+"obs_t_others_agt_" + str(self.ID))
        self.obs_next_others = tf.placeholder(tf.float32, [None, self.dim_o * self.n_others],
                                              name=self.label+"obs_next_others_agt_" + str(self.ID))
        self.act_others = tf.placeholder(tf.float32, [None, self.dim_a * self.n_others],
                                         name=self.label+"act_t_others_agt_" + str(self.ID))
        self.act_next_others = tf.placeholder(tf.float32, [None, self.dim_a * self.n_others],
                                              name=self.label+"act_next_others_agt_" + str(self.ID))
        self.act_teacher = tf.placeholder(tf.float32, [None, self.dim_a], self.label+"act_teacher_agt_" + str(self.ID))
        # Input And Output for Critic Network
        self.Q = self.critic_net(obs_in=self.obs_t, act_in=self.act_t,
                                 obs_others_in=self.obs_others, act_others_in=self.act_others,
                                 scope=self.label+"critic_net_agt_" + str(self.ID))
        self.Q_next = self.critic_net(obs_in=self.obs_next, act_in=self.act_next,
                                      obs_others_in=self.obs_next_others, act_others_in=self.act_next_others,
                                      scope=self.label+"critic_target_net_agt_" + str(self.ID))
        self.Q_teacher = tf.placeholder(tf.float32, [None, 1], self.label+"Q_teacher_value_agt_" + str(self.ID))
        self.reward = tf.placeholder(tf.float32, [None, 1], name=self.label+"reward_t_agt_" + str(self.ID))
        self.Q_predict = tf.placeholder(tf.float32, [None, 1], name=self.label+"q_predict_agt_" + str(self.ID))
        # self.Q_predict = self.reward + self.gamma * self.Q_next

        # Parameter Collection
        self.param_a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label+"actor_net_agt_" + str(self.ID))
        self.param_a_tar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope=self.label+"actor_target_net_agt_" + str(self.ID))
        self.param_c = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.label+"critic_net_agt_" + str(self.ID))
        self.param_c_tar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope=self.label+"critic_target_net_agt_" + str(self.ID))

        # loss function and Optimizer
        self.TD_error = self.Q - self.Q_predict
        self.loss_c_origin = tf.reduce_mean(tf.square(self.TD_error))
        self.loss_c_distill = self.loss_distilled(teacher=self.Q_teacher, student=self.Q,
                                                  loss_function="MSE", name=self.label+"loss_critic_distilled")
        self.loss_a_origin = tf.reduce_mean(-self.Q)
        self.loss_a_distill = self.loss_distilled(teacher=self.act_teacher, student=self.act_t,
                                                  loss_function="MSE", name=self.label+"loss_actor_distilled")
        if self.ID <= self.n_old_pre:
            self.loss_a = self.loss_a_origin  # + self.weight_distill * self.loss_a_distill
            self.loss_c = self.loss_c_origin  # + self.weight_distill * self.loss_c_distill
        else:
            self.loss_a = self.loss_a_origin + self.weight_distill * self.loss_a_distill
            self.loss_c = self.loss_c_origin + self.weight_distill * self.loss_c_distill

        self.trainer_a = tf.train.AdamOptimizer(self.lr_actor).minimize(self.loss_a, var_list=self.param_a)
        self.trainer_c = tf.train.AdamOptimizer(self.lr_critic).minimize(self.loss_c, var_list=self.param_c)

        # soft update for target network
        self.soft_update_a = [self.param_a_tar[i].assign(
            tf.multiply(self.param_a[i], self.tau) + tf.multiply(self.param_a_tar[i], 1 - self.tau)) for i in
            range(len(self.param_a_tar))]
        for i in range(len(self.param_a_tar)):
            self.soft_update_a[i] = tf.assign(self.param_a_tar[i],
                                              tf.multiply(self.param_a[i], self.tau) + tf.multiply(self.param_a_tar[i],
                                                                                                   1 - self.tau))
        self.soft_update_c = [self.param_c_tar[i].assign(
            tf.multiply(self.param_c[i], self.tau) + tf.multiply(self.param_c_tar[i], 1 - self.tau)) for i in
            range(len(self.param_c_tar))]
        for i in range(len(self.param_c_tar)):
            self.soft_update_c[i] = tf.assign(self.param_c_tar[i],
                                              tf.multiply(self.param_c[i], self.tau) + tf.multiply(self.param_c_tar[i],
                                                                                                   1 - self.tau))
        # if self.ID > (self.n_agents - args.n_newers):
        #     self.trainer_transfer_a = tf.train.AdamOptimizer(self.lr_tranfer).minimize(self.loss_a_distill,
        #                                                                                var_list=self.param_a)
        #     self.trainer_transfer_c = tf.train.AdamOptimizer(self.lr_tranfer).minimize(self.loss_c_distill,
        #                                                                                var_list=self.param_c)

    def actor_net(self, obs_in, scope):
        with tf.variable_scope(scope):
            x_in = obs_in
            # hidden layers
            for idx_layer in range(self.dim_units_a.__len__()):
                layer = tf.layers.dense(
                    inputs=x_in,
                    units=self.dim_units_a[idx_layer],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer_' + str(idx_layer)
                )
                x_in = layer

            # output layer
            output_a = tf.layers.dense(
                inputs=x_in,
                units=self.dim_a,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001),
                bias_initializer=tf.constant_initializer(0),
                name='layer_output'
            )

            return output_a

    def critic_net(self, obs_in, act_in, obs_others_in, act_others_in, scope):
        with tf.variable_scope(scope):
            # hidden layers
            x_in = tf.concat([obs_in, act_in, obs_others_in, act_others_in], axis=1)
            for idx_layer in range(self.dim_units_c.__len__()):
                layer = tf.layers.dense(
                    inputs=x_in,
                    units=self.dim_units_a[idx_layer],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.03),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer_' + str(idx_layer)
                )
                x_in = layer

            # output layer
            output_q = tf.layers.dense(
                inputs=x_in,
                units=1,
                activation=None,
                kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001),
                bias_initializer=tf.constant_initializer(0.1),
                name='layer_output'
            )

            return output_q

    def update_target_net(self, sess, init=False):
        sess.run(self.soft_update_a)
        sess.run(self.soft_update_c)

        if init:
            for i in range(len(self.param_c_tar)):
                sess.run(tf.assign(self.param_c_tar[i], self.param_c[i]))
            for i in range(len(self.param_a_tar)):
                sess.run(tf.assign(self.param_a_tar[i], self.param_a[i]))

    def get_action(self, observation, sess, noise=False):
        action_t = self.act_t.eval(feed_dict={self.obs_t: observation}, session=sess)
        if noise:
            self.mu = action_t
            for i in range(self.dim_a):
                action_t[:, i] = action_t[:, i] + np.random.normal(0, self.sigma[i][i])

        return action_t

    def get_q_values(self, obs, obs_others, act, act_others, sess):
        return self.Q.eval(feed_dict={self.obs_t: obs,
                                      self.obs_others: obs_others,
                                      self.act_t: act,
                                      self.act_others: act_others},
                           session=sess)

    def get_q_predict(self, r, obs_next, obs_next_others, act_next_others, sess):
        q_next = self.Q_next.eval(feed_dict={self.obs_next: obs_next,
                                             self.obs_next_others: obs_next_others,
                                             self.act_next_others: act_next_others},
                                  session=sess)
        q_predict = r + self.gamma * q_next

        return q_predict

    def loss_distilled(self, teacher, student, loss_function=None, name=None):
        with tf.name_scope(name):
            if loss_function == "MSE":
                loss_dist = tf.reduce_mean(tf.square(teacher - student))
            elif loss_function == "MAE":
                loss_dist = tf.reduce_mean(tf.abs(teacher - student))
            else:
                loss_dist = None
        return loss_dist
