import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # num_agents = 3
        num_agents = 5
        world.num_agents = num_agents
        self.num_adversaries = 1

        self.kill_reward = 10
        self.num_goals = 2
        # Need this to access in environment.py
        world.num_goals = self.num_goals
        self.goal_reward = 5

        # num_landmarks = num_agents - 1
        num_landmarks = 10
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        self.one_time_reward = True

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < self.num_adversaries else False
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        self.scenario_dead_agents = 0
        # random properties for agents
        for i in range(0, self.num_adversaries):
            world.agents[i].color = np.array([0.85, 0.35, 0.35])
        # world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(self.num_adversaries, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        # goal = np.random.choice(world.landmarks)
        # goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            goals = np.random.choice(world.landmarks, self.num_goals, False)
            agent.goals_visited = np.full(self.num_goals, False)
            agent.goals = goals
            #CUSTOM reset
            agent.alive = True
            for goal in goals:
                goal.color = np.array([0.15, 0.65, 0.15])
            # goal.color = np.array([0.15, 0.65, 0.15])
            # agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = False
        shaped_adv_reward = False

        # Calculate negative reward for adversary
        # adversary_agents = self.adversaries(world)
        # if shaped_adv_reward:  # distance-based adversary reward
        #     adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        # else:  # proximity-based adversary reward (binary)
        #     adv_rew = 0
        #     for a in adversary_agents:
        #         if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
        #             adv_rew -= 5
        adv_rew = sum([np.sqrt(np.sum(np.square(agent.state.p_pos - adversary.state.p_pos))) for adversary in self.adversaries(world)])
        # adv_rew = -sum([self.adversary_reward(adversary, world) for adversary in self.adversaries(world)])

        # Calculate positive reward for agents
        pos_rew = 0
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            ## Changed the shared reward default to also account for multiple goals
            ## Previously code took the negative min of distances.
            ## Replaced with max, since calc_goal_reward already negates the distances

            pos_rew = max([self.calc_goal_reward(a) for a in good_agents])

            # pos_rew = -min(
            #     [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        else:
            if agent.alive:
                pos_rew += self.kill_reward

            pos_rew += self.calc_goal_reward(agent)

            # pos_rew -= min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        # else:  # proximity-based agent reward (binary)
        #     pos_rew = 0
        #     if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
        #             < 2 * agent.goal_a.size:
        #         pos_rew += 5
        #     pos_rew -= min(
        #         [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])

        return pos_rew + adv_rew

    def calc_goal_reward(self, agent):
        rew = 0
        for idx in range(self.num_goals):
            # agent has visited this goal
            if agent.goals_visited[idx]:
                rew += self.goal_reward
            # else, penalize agent reward based on distance to goal
            else:
                rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goals[idx].state.p_pos)))
        return rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark

        shaped_reward = False
        if shaped_reward:  # distance-based reward
            # TODO this base case will not work for multi-goal
            # not sure why it considers adv goal dist in the first place
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            adv_rew = 0
            # kill reward (continuously reward adversary for dead agents)
            # TODO: revisit this to try out one-time kill reward
            adv_rew += self.kill_reward * world.dead_agents
            # distance to closest living agent
            alive_good_agents = [a for a in self.good_agents(world) if a.alive]
            if alive_good_agents:
                adv_rew -= min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in alive_good_agents])
            return adv_rew
        # else:  # proximity-based reward (binary)
        #     adv_rew = 0
        #     if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
        #         adv_rew += 5
        #     return adv_rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            # return -> [v_self_goal, |v_other_agents_goal|, |v_self_other_agents|]
            return np.concatenate([g.state.p_pos - agent.state.p_pos for g in agent.goals] + entity_pos + other_pos)
        else:
            # return -> [|v_other_agents_goal|, |v_self_other_agents|]
            return np.concatenate(entity_pos + other_pos)
