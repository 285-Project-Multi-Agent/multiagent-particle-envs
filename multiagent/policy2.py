import numpy as np
from pyglet.window import key

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        epsilon = 0.001
        self.agent_index = agent_index
        self.directions = [
            np.array([-epsilon, 0]),
            np.array([epsilon, 0]),
            np.array([0, epsilon]),
            np.array([0, -epsilon])
        ]
        self.fived = {0: 1,
        1:2,
        3:3,
        2:4}
    def action(self, obs):
        # ignore observation and just act based on keyboard events
        agent = np.array([obs[self.agent_index], obs[self.agent_index + 1]])
        target = np.array([obs[self.agent_index + 2], obs[self.agent_index + 3]])
        if self.env.discrete_action_input:
            u = self.fived[np.argmin([np.linalg.norm(target - (agent + self.directions[i])) for i in range(4)])]
            return u
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[self.fived[np.argmin([np.linalg.norm(target - (agent + self.directions[i])) for i in range(4)])]] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
