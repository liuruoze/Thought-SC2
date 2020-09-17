from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import tensorflow as tf
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

from lib import utils as U
from lib import config as C

from lib import transform_pos as T
from lib import option as M
from lib import environment
from lib import my_sc2_env as sc2_env
from lib.replay_buffer import Buffer
from mini_agent import ProtossAction

from mini_network_dream import SecondNetwork

class MiniSourceAgent(base_agent.BaseAgent):
    """Agent for source game of starcraft."""

    def __init__(self, index=0, rl_training=False, restore_model=False, global_buffer=None, net=None, sec_net=None,
        strategy_agent=None, greedy_action=False,
        extract_save_dir=None):
        super(MiniSourceAgent, self).__init__()
        self.net = net
        self.sec_net = sec_net
        self.index = index
        self.global_buffer = global_buffer
        self.restore_model = restore_model

        # count num
        self.step = 0

        self.strategy_wait_secs = 2
        self.strategy_flag = False
        self.policy_wait_secs = 2
        self.policy_flag = True

        self.env = None
        self.obs = None

        # buffer
        self.local_buffer = Buffer()

        self.num_players = 2
        self.on_select = None
        self._result = None
        self._gases = None
        self.is_end = False

        self.rl_training = rl_training

        self.rnn_state = None
        self.zero_state = self.sec_net.rnn_init_state()

        self.extract_save_dir = extract_save_dir
        self.reset()

    def reset(self):
        super(MiniSourceAgent, self).reset()
        self.step = 0
        self.obs = None
        self._result = None
        self._gases = None
        self.is_end = False

        self.strategy_flag = False
        self.policy_flag = True

        self.local_buffer.reset()

        self.rnn_state = self.zero_state

    def set_env(self, env):
        self.env = env

    def init_network(self):
        self.net.initialize()
        if self.restore_model:
            self.net.restore_policy()

    def reset_old_network(self):
        self.net.reset_old_network()

    def save_model(self):
        self.net.save_policy()

    def update_policy(self):
        self.net.Update_policy(self.global_buffer)

    def update_result(self, result_list):
        self.net.update_result(result_list)

    def update_network(self, result_list):
        self.net.Update_policy(self.global_buffer)
        self.net.Update_result(result_list)

    def update_summary(self, counter):
        return self.net.Update_summary(counter)

    def mini_step(self, action):
        if action == ProtossAction.Build_probe.value:
            M.mineral_worker(self)

        elif action == ProtossAction.Build_zealot.value:
            M.train_army(self, C._TRAIN_ZEALOT)

        elif action == ProtossAction.Build_Stalker.value:
            M.train_army(self, C._TRAIN_STALKER)

        elif action == ProtossAction.Build_pylon.value:
            no_unit_index = U.get_unit_mask_screen(self.obs, size=2)
            pos = U.get_pos(no_unit_index)
            M.build_by_idle_worker(self, C._BUILD_PYLON_S, pos)

        elif action == ProtossAction.Build_gateway.value:
            power_index = U.get_power_mask_screen(self.obs, size=5)
            pos = U.get_pos(power_index)
            M.build_by_idle_worker(self, C._BUILD_GATEWAY_S, pos)

        elif action == ProtossAction.Build_Assimilator.value:
            if self._gases is not None:
                #U.find_gas_pos(self.obs, 1)
                gas_1 = self._gases[0]
                gas_2 = self._gases[1]

                if gas_1 is not None and not U.is_assimilator_on_gas(self.obs, gas_1):
                    gas_1_pos = T.world_to_screen_pos(self.env.game_info, gas_1.pos, self.obs)
                    M.build_by_idle_worker(self, C._BUILD_ASSIMILATOR_S, gas_1_pos)

                elif gas_2 is not None and not U.is_assimilator_on_gas(self.obs, gas_2):
                    gas_2_pos = T.world_to_screen_pos(self.env.game_info, gas_2.pos, self.obs)
                    M.build_by_idle_worker(self, C._BUILD_ASSIMILATOR_S, gas_2_pos)

        elif action == ProtossAction.Build_CyberneticsCore.value:
            power_index = U.get_power_mask_screen(self.obs, size=3)
            pos = U.get_pos(power_index)
            M.build_by_idle_worker(self, C._BUILD_CYBER_S, pos)

        elif action == ProtossAction.Attack.value:
            M.attack_step(self)

        elif action == ProtossAction.Retreat.value:
            M.retreat_step(self)

        elif action == ProtossAction.Do_nothing.value:
            self.safe_action(C._NO_OP, 0, [])

    def get_the_input(self):
        high_input, tech_cost, pop_num = U.get_input(self.obs)
        controller_input = np.concatenate([high_input, tech_cost, pop_num], axis=0)
        return controller_input

    def mapping_source_to_mini_by_rule(self, source_state):
        simple_input = np.zeros([20], dtype=np.int16)
        simple_input[0] = 0  # self.time_seconds 
        simple_input[1] = source_state[28]  # self.mineral_worker_nums
        simple_input[2] = source_state[30] + source_state[32]  # self.gas_worker_nums
        simple_input[3] = source_state[2]  # self.mineral
        simple_input[4] = source_state[3]  # self.gas
        simple_input[5] = source_state[6]  # self.food_cup
        simple_input[6] = source_state[7]  # self.food_used
        simple_input[7] = source_state[10]  # self.army_nums

        simple_input[8] = source_state[16]  # self.gateway_num
        simple_input[9] = source_state[14]  # self.pylon_num
        simple_input[10] = source_state[15]  # self.Assimilator_num
        simple_input[11] = source_state[17]  # self.CyberneticsCore_num

        simple_input[12] = source_state[12]  # self.zealot_num
        simple_input[13] = source_state[13]  # self.Stalker_num
        simple_input[14] = source_state[11]  # self.probe_num

        simple_input[15] = source_state[4] + source_state[2]  # self.collected_mineral
        simple_input[16] = source_state[4]  # self.spent_mineral
        simple_input[17] = source_state[5] + source_state[3]  # self.collected_gas
        simple_input[18] = source_state[5]  # self.spent_gas
        simple_input[19] = 1  # self.Nexus_num

        return simple_input

    def play(self, verbose=False):
        self.play_train(verbose=verbose)

    def sample(self, verbose=False, use_image=True):
        is_attack = False
        state_last = None

        random_generated_int = random.randint(0, 2**31-1)
        filename = self.extract_save_dir+"/"+str(random_generated_int)+".npz"
        
        recording_obs = []
        recording_img = []
        recording_action = []
        recording_reward = []

        np.random.seed(random_generated_int)
        tf.set_random_seed(random_generated_int)
        
        self.safe_action(C._NO_OP, 0, [])
        self.safe_action(C._MOVE_CAMERA, 0, [C.base_camera_pos])
        self._gases = U.find_initial_gases(self.obs)
        while True:

            self.safe_action(C._MOVE_CAMERA, 0, [C.base_camera_pos])
            if self.policy_flag and (not self.is_end):

                non_image_feature = self.mapping_source_to_mini_by_rule(self.get_the_input())
                #print('non_image_feature.shape:', non_image_feature.shape)
                #print('non_image_feature:', non_image_feature)

                image_feature = U.get_simple_map_data(self.obs)
                #print('image_feature.shape:', image_feature.shape)
                #print('image_feature:', image_feature)

                latent_image_feature, mu, logvar = self.encode_obs(image_feature)
                #print('latent_image_feature.shape:', latent_image_feature.shape)
                #print('latent_image_feature:', latent_image_feature)

                feature = np.concatenate([non_image_feature, latent_image_feature], axis=-1)
                #print('feature.shape:', feature.shape)
                #print('feature:', feature)

                #state_now = feature
                reward_last = 0
                state_now, action, v_preds = self.get_action(feature, reward_last)

                # print(ProtossAction(action).name)
                self.mini_step(action)

                if state_last is not None:
                    if False:
                        print('state_last:', state_last, ', action_last:', action_last, ', state_now:', state_now)
                    v_preds_next = self.net.policy.get_values(state_now)
                    v_preds_next = self.get_values(v_preds_next)
                    reward = 0
                    
                    recording_obs.append(non_image_feature)
                    recording_img.append(image_feature)
                    recording_action.append(action)
                    recording_reward.append(reward)

                    #self.local_buffer.append(state_last, action_last, state_now, reward, v_preds, v_preds_next)

                state_last = state_now
                action_last = action
                self.policy_flag = False

            if self.is_end:
                if True:                   
                    # consider the win/loss, to 0(not end), 1(loss), 2(draw), 3(win)
                    recording_reward[-1] = (1 * self.result['reward'] + 2)
                    if recording_reward[-1] != 0:
                        print("result is:", recording_reward[-1])


                    recording_obs = np.array(recording_obs, dtype=np.uint16)
                    recording_action = np.array(recording_action, dtype=np.uint8)
                    recording_reward = np.array(recording_reward, dtype=np.uint8)
                    recording_img = np.array(recording_img, dtype=np.float16)
                    
                    np.savez_compressed(filename, obs=recording_obs, img=recording_img, action=recording_action, reward=recording_reward)
                break


    def play_train(self, continues_attack=False, verbose=False):
        is_attack = False
        state_last = None

        self.safe_action(C._NO_OP, 0, [])
        self.safe_action(C._MOVE_CAMERA, 0, [C.base_camera_pos])
        self._gases = U.find_initial_gases(self.obs)

        while True:

            self.safe_action(C._MOVE_CAMERA, 0, [C.base_camera_pos])
            if self.policy_flag and (not self.is_end):

                non_image_feature = self.mapping_source_to_mini_by_rule(self.get_the_input())
                #print('non_image_feature.shape:', non_image_feature.shape)
                #print('non_image_feature:', non_image_feature)

                image_feature = U.get_simple_map_data(self.obs)
                #print('image_feature.shape:', image_feature.shape)
                #print('image_feature:', image_feature)

                latent_image_feature, mu, logvar = self.encode_obs(image_feature)
                #print('latent_image_feature.shape:', latent_image_feature.shape)
                #print('latent_image_feature:', latent_image_feature)

                feature = np.concatenate([non_image_feature, latent_image_feature], axis=-1)
                #print('feature.shape:', feature.shape)
                #print('feature:', feature)

                #state_now = feature
                reward_last = 0
                state_now, action, v_preds = self.get_action(feature, reward_last)

                # print(ProtossAction(action).name)
                self.mini_step(action)

                if state_last is not None:
                    if 0:
                        print('state_last:', state_last, ', action_last:', action_last, ', state_now:', state_now)
                    v_preds_next = self.net.policy.get_values(state_now)
                    v_preds_next = self.get_values(v_preds_next)
                    reward = 0
                    self.local_buffer.append(state_last, action_last, state_now, reward, v_preds, v_preds_next)

                state_last = state_now
                action_last = action
                self.policy_flag = False

            if self.is_end:
                if self.rl_training:
                    self.local_buffer.rewards[-1] += 1 * self.result['reward']  # self.result['win']
                    #print(self.local_buffer.rewards)
                    self.global_buffer.add(self.local_buffer)
                    #print("add %d buffer!" % (len(self.local_buffer.rewards)))
                break


    def encode_obs(self, obs):
        # convert raw obs to z, mu, logvar
        result = np.copy(obs)
        result = result.reshape(1, 64, 64, 12)
        mu, logvar = self.sec_net.vae.encode_mu_logvar(result)
        mu = mu[0]
        logvar = logvar[0]
        s = logvar.shape
        z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
        return z, mu, logvar


    def get_action(self, feature, reward):
        input_h = self.sec_net.rnn_output(self.rnn_state, feature)

        action, v_preds = self.net.policy.get_action(input_h, verbose=False)

        self.rnn_state = self.sec_net.rnn_next_state(feature, action, reward, self.rnn_state)
        return input_h, action, v_preds


    def set_flag(self):
        if self.step % C.time_wait(self.strategy_wait_secs) == 1:
            self.strategy_flag = True

        if self.step % C.time_wait(self.policy_wait_secs) == 1:
            self.policy_flag = True

    def safe_action(self, action, unit_type, args):
        if M.check_params(self, action, unit_type, args, 1):
            obs = self.env.step([sc2_actions.FunctionCall(action, args)])[0]
            self.obs = obs
            self.step += 1
            self.update_result()
            self.set_flag()

    def select(self, action, unit_type, args):
        # safe select
        if M.check_params(self, action, unit_type, args, 0):
            self.obs = self.env.step([sc2_actions.FunctionCall(action, args)])[0]
            self.on_select = unit_type
            self.update_result()
            self.step += 1
            self.set_flag()

    @property
    def result(self):
        return self._result

    def update_result(self):
        if self.obs is None:
            return
        if self.obs.last() or self.env.state == environment.StepType.LAST:
            self.is_end = True
            outcome = 0
            o = self.obs.raw_observation
            player_id = o.observation.player_common.player_id
            for r in o.player_result:
                if r.player_id == player_id:
                    outcome = sc2_env._possible_results.get(r.result, 0)
            frames = o.observation.game_loop
            result = {}
            result['outcome'] = outcome
            result['reward'] = self.obs.reward
            result['frames'] = frames

            self._result = result
            print('play end, total return', self.obs.reward)
            self.step = 0

    def get_values(self, values):
        # check if the game is end
        if self.is_end and self.result['reward'] != 0:
            return 0
        else:
            return values
