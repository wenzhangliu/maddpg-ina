import numpy as np
from copy import deepcopy


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.05
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = 1.2
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        self.real_mass = 1.0

    @property
    def mass(self):
        mass = self.initial_mass * np.float(4/3) * np.pi * np.power(self.size,3) * self.density
        return mass


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()



# properties of box entities
class Box(Entity):
    def __init__(self):
        super(Box, self).__init__()
        self.movable = True
        self.blind = True

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = None
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.boxes = []
        self.num_landmarks = 0
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.3
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # lanmark move range
        self.move_range_agent = 1.0
        self.move_range_landmark = 1.0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.boxes

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        p_force = [None] * len(self.entities)  # gather forces applied to entities
        p_force = self.apply_action_force(p_force)  # apply agent physical controls
        p_force = self.apply_environment_force(p_force)  # apply environment forces
        self.integrate_state(p_force)  # integrate physical state

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.agents):
            for b, entity_b in enumerate(self.agents):
                if (b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.agents):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                if entity.u_range is not None:
                    force_a = np.sqrt(np.square(p_force[i][0]) + np.square(p_force[i][1]))
                    if force_a > entity.u_range:
                        p_force[i] = p_force[i] / force_a * entity.u_range
                entity.state.p_vel += (p_force[i] / entity.real_mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

        if agent.wall:
            if (agent.state.p_pos[0] + agent.size) > self.move_range_agent:
                agent.state.p_pos[0] = 2 * (self.move_range_agent-0.8*agent.size) - agent.state.p_pos[0]
                agent.state.p_vel[0] = - agent.state.p_vel[0]
                # agent.action.u[0] = - agent.action.u[0]
                agent.action.u[0] = 0
            if (agent.state.p_pos[0] - agent.size) < -self.move_range_agent:
                agent.state.p_pos[0] = -2 * (self.move_range_agent-0.8*agent.size) - agent.state.p_pos[0]
                agent.state.p_vel[0] = - agent.state.p_vel[0]
                # agent.action.u[0] = - agent.action.u[0]
                agent.action.u[0] = 0
            if (agent.state.p_pos[1] + agent.size) > self.move_range_agent:
                agent.state.p_pos[1] = 2 * (self.move_range_agent-0.8*agent.size) - agent.state.p_pos[1]
                agent.state.p_vel[1] = - agent.state.p_vel[1]
                # agent.action.u[1] = - agent.action.u[1]
                agent.action.u[0] = 0
            if (agent.state.p_pos[1] - agent.size) < -self.move_range_agent:
                agent.state.p_pos[1] = -2 * (self.move_range_agent-0.8*agent.size) - agent.state.p_pos[1]
                agent.state.p_vel[1] = - agent.state.p_vel[1]
                # agent.action.u[1] = - agent.action.u[1]
                agent.action.u[0] = 0

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        if dist < dist_min:
            # softmax penetration
            k = self.contact_margin
            penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
            force = self.contact_force * delta_pos / dist * penetration * 0.1
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
            return [force_a, force_b]
        else:
            force_a = 0
            force_b = 0
            return [force_a, force_b], False

    def update_box(self, box):
        # set communication state (directly for now)
        if box.silent:
            box.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*box.action.c.shape) * box.c_noise if box.c_noise else 0.0
            box.state.c = box.action.c + noise

        if box.wall:
            if (box.state.p_pos[0] + box.size) > self.move_range_agent:
                box.state.p_pos[0] = 2 * (self.move_range_agent-box.size) - box.state.p_pos[0]
                box.state.p_vel[0] = - box.state.p_vel[0]
                # box.action.u[0] = box.action.u[0] * 0.1
            if (box.state.p_pos[0] - box.size) < -self.move_range_agent:
                box.state.p_pos[0] = -2 * (self.move_range_agent-box.size) - box.state.p_pos[0]
                box.state.p_vel[0] = - box.state.p_vel[0]
                # box.action.u[0] = box.action.u[0] * 0.1
            if (box.state.p_pos[1] + box.size) > self.move_range_agent:
                box.state.p_pos[1] = 2 * (self.move_range_agent-box.size) - box.state.p_pos[1]
                box.state.p_vel[1] = - box.state.p_vel[1]
                # box.action.u[1] = box.action.u[1] * 0.1
            if (box.state.p_pos[1] - box.size) < -self.move_range_agent:
                box.state.p_pos[1] = -2 * (self.move_range_agent-box.size) - box.state.p_pos[1]
                box.state.p_vel[1] = - box.state.p_vel[1]
                # box.action.u[1] = box.action.u[1] * 0.1

    def agent_box_collision(self):
        pos_next = np.zeros([self.agents.__len__(), 2])
        vel_next = np.zeros([self.agents.__len__(), 2])
        force_box = np.zeros([self.agents.__len__(), 2])
        for a, agent in enumerate(self.agents):
            for b, box in enumerate(self.boxes):
                delta_pos = agent.state.p_pos - box.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                dist_min = agent.size + box.size
                if dist > dist_min:
                    box.collide_with_agents[int(agent.name[-1])] = False
                    return
                box.collide_with_agents[int(agent.name[-1])] = True
                ## calculate the positions and velosity after collision for agents
                d_square = np.square(delta_pos)
                dxdy = delta_pos[0] * delta_pos[1]
                d_square_sum = d_square[0] + d_square[1]
                d_square_dif = d_square[0] - d_square[1]

                ## get collide forces for box
                k = self.contact_margin
                penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
                force_box[a] = -self.contact_force * delta_pos / dist * penetration
                ## get next velosities for agent
                if 0.0 == d_square_sum:
                    continue
                vel_next[a, 0] = (-d_square_dif * agent.state.p_vel[0] - 2 * dxdy * agent.state.p_vel[1]) / d_square_sum
                vel_next[a, 1] = (d_square_dif * agent.state.p_vel[1] - 2 * dxdy * agent.state.p_vel[0]) / d_square_sum
                agent.state.p_vel = vel_next[a]
                ## get next positions for agent
                if box.size > dist:
                    beyond_dist = 2 * (box.size - dist)
                else:
                    beyond_dist = 2 * (agent.size + box.size - dist)
                sin_theta = delta_pos[1] / np.sqrt(d_square_sum)
                cos_theta = delta_pos[0] / np.sqrt(d_square_sum)
                pos_next[a, 0] = agent.state.p_pos[0] + beyond_dist * cos_theta
                pos_next[a, 1] = agent.state.p_pos[1] + beyond_dist * sin_theta
                agent.state.p_pos = pos_next[a]
        ## update box states
        for b, box in enumerate(self.boxes):
            if all(box.collide_with_agents) is True:
                box.state.p_vel = box.state.p_vel * (1 - self.damping)
                force = np.sum(force_box, axis=0)
                box.state.p_vel += (force / box.real_mass) * self.dt
                box.state.p_pos += box.state.p_vel * self.dt

        return

