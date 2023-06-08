import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 1
        world.num_landmarks = num_landmarks
        self.num_landmarks = num_landmarks
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            # agent.size = 0.15
            agent.size = 0.08
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = [1.0, 0.0, 0.0]
        world.agents[1].color = [0.0, 0.0, 1.0]
        world.landmarks[0].color = [0.0, 0.0, 1.0]

        # set random initial states
        for agent in world.agents:
            # agent.state.p_pos = np.random.uniform(-1.0, +1.0, world.dim_p)
            agent.state.p_pos = np.zeros(world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        r_attactor = np.random.uniform(0.6, +1.5)
        angle = np.random.uniform(-math.pi, math.pi)
        world.agents[0].state.p_pos = [r_attactor * math.cos(angle), r_attactor * math.sin(angle)]
        world.agents[1].state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.zeros(world.dim_p) # np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist <= dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = np.zeros(self.num_landmarks+1)

        if agent.name == "agent 0": # attacker
            rew[0] = -np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))
            if self.is_collision(agent, world.agents[1]):
                rew[1] = -10

        if agent.name == "agent 1": # intercepter
            # rew[0] = -np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[0].state.p_pos)))
            rew[0] = -1
            if self.is_collision(agent, world.agents[0]):
                rew[0] = 0

            if self.is_collision(agent, world.landmarks[0]):
                rew[1] = -10

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
