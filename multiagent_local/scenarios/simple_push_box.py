import numpy as np
from multiagent_local.core import World, Agent, Landmark, Box
from multiagent_local.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_old=2, num_new=0):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_old + num_new
        num_boxes = 1
        num_adversaries = 0
        num_landmarks = 1
        self.size_agent = 0.03
        self.size_landmark = 0.05
        self.size_box = 0.30
        density_agent = 500
        density_landmark = 40
        density_box = 250
        self.move_range = 8.0
        # add agents
        # world.collide_penalty = True
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.size = self.size_agent
            agent.density = density_agent
            agent.real_mass = agent.mass
            agent.collide = True
            agent.silent = True
            agent.wall = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
        # add boxes
        world.boxes = [Box() for i in range(num_boxes)]
        for i, box in enumerate(world.boxes):
            box.name = 'box %d' % i
            box.size = self.size_box
            box.density = density_box
            box.real_mass = box.mass
            box.collide = True
            box.silent = True
            box.wall = True
            box.collide_with_agents = [False for i in range(num_agents)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.size = self.size_landmark
            landmark.density = density_landmark
            landmark.real_mass = landmark.mass
            landmark.collide = False
            landmark.movable = False
            landmark.wall = True
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.5, 0.5, 0.5])
            # landmark.color[i + 1] += 0.8
            landmark.index = i
        for i, box in enumerate(world.boxes):
            box.color = np.array([0.7, 0.8, 0.8])
            box.index = i
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            agent.color = np.array([0.20*(i+1), 0.20*(i+1), 0.20*(i+1)])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                j = goal.index
                agent.color[j + 1] += 0.5
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1 + self.size_box, 1 - self.size_box, world.dim_p)
            # landmark.state.p_pos = np.array([1 - self.size_box - self.size_agent, 1-self.size_box-self.size_agent])
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, box in enumerate(world.boxes):
            box.state.p_pos = np.random.uniform(-1+box.size, 1-box.size, world.dim_p)
            # box.state.p_pos = np.array([-1 + box.size + 0.5, -1+box.size + 0.5])
            box.state.p_vel = np.zeros(world.dim_p)
        for agent in world.agents:
            wrong_pos = True
            while wrong_pos:
                agent.state.p_pos = np.random.uniform(-1+agent.size, +1-agent.size, world.dim_p)
                d_agt_box = np.sqrt(np.sum(np.square(agent.state.p_pos - world.boxes[0].state.p_pos)))
                if d_agt_box >= (agent.size + world.boxes[0].size):
                    wrong_pos = False
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # the distance to the goal
        # rew = 0.0
        rew = -0.1
        base_reward = 0.0
        for i, box in enumerate(world.boxes):
            base_reward += (-2.0 * np.sqrt(np.sum(np.square(box.state.p_pos - agent.goal_a.state.p_pos))))
        # if world.base_reward is True:
        #     return base_reward
        rew += base_reward
        # collide with other agents
        if agent.collide:
            # if world.collide_penalty is True:
            for a in world.agents:
                if self.is_collision(a, agent) and (a.name != agent.name):
                    rew -= 0.1
            # collide in the wall will get penalty
            if self.collide_wall(agent):
                rew -= 0.1
        for box in world.boxes:
            if self.is_collision(agent, box):
                rew += 0.1  # 1.0

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        landmark_pos = []
        for entity in world.landmarks:  # world.entities:
            landmark_pos.append(entity.state.p_pos - agent.state.p_pos)
        # get positions and velocity of all entities in this agent's reference frame
        box_pos, box_vel = [], []
        for entity in world.boxes:
            box_pos.append(entity.state.p_pos - agent.state.p_pos)
            box_vel.append(entity.state.p_vel - agent.state.p_vel)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        if not agent.adversary:
            return np.concatenate(
                [agent.state.p_vel] + landmark_pos + box_pos + other_pos)
            # return np.concatenate(
            #     [agent.state.p_vel] + landmark_pos + box_pos + box_vel + other_pos)
        else:
            # other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate([agent.state.p_vel] + landmark_pos + box_pos + box_vel + other_pos)

    def collide_wall(self, agent):
        position = agent.state.p_pos
        if (abs(position[0]) > 1.0) or (abs(position[1]) > 1.0):
            return True
        else:
            return False

    def done(self, agent):
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos)))
        if dist > self.move_range:
            return True
        else:
            return False