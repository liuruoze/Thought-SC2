import numpy as np
import copy
import time


class MPC:

    def __init__(self, MAX_ACTIONS):
        self.MAX_ACTIONS = MAX_ACTIONS
        self.K = 5000   # num_of_sample_actions
        self.H = 30     # horizon_length

    def get_action(self, state, agent_clone, verbose=False):
        agent_clone = copy.deepcopy(agent_clone)
        action, r_preds = self.shooting(state_now=state, agent_clone=agent_clone, num_of_sample_actions=self.K,
                                        horizon_length=self.H, MAX_ACTIONS=self.MAX_ACTIONS)
        if True:
            print('action:', action)
            print('r_preds:', r_preds)

        return action, r_preds

    def shooting(self, state_now, agent_clone, num_of_sample_actions, horizon_length, MAX_ACTIONS):
        max_cumulative_reward = float('-inf')
        optimal_act_seq = np.array([])
        for i in range(num_of_sample_actions):
            run_agent = copy.deepcopy(agent_clone)
            rand_act_seq = np.random.randint(MAX_ACTIONS, size=horizon_length)
            state = state_now
            cumulative_reward = 0
            for j in range(horizon_length):
                act = rand_act_seq[j]
                state_next = run_agent.get_next_state(act)
                if 0:
                    print('state now:', state_next.astype(dtype=np.int32))
                    time.sleep(1)
                r = run_agent.get_mineral_reward(state, state_next)
                state = state_next
                cumulative_reward += r
            #print('cumulative_reward:', cumulative_reward)
            if cumulative_reward > max_cumulative_reward:
                max_cumulative_reward = cumulative_reward
                optimal_act_seq = rand_act_seq

        # because we use mpc, so we only need to execuate the first action
        execuate = optimal_act_seq[0]
        return execuate, max_cumulative_reward
