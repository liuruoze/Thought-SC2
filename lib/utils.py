import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
#from skimage.transform import resize
from skimage.transform import rescale, resize

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
import lib.transform_pos as T
from lib import config as C
import param as P

_MINIMAP_SELECTED = features.MINIMAP_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index
_VISIABLE_MAP = features.SCREEN_FEATURES.visibility_map.index
_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_SELF_INDEX = 1
_VISIBLE_INDEX = 1
_ENEMY_INDEX = 4
_PROBE_TYPE_INDEX = 84
_PYLON_TYPE_INDEX = 60
_FORGE_TYPE_INDEX = 63
_CANNON_TYPE_INDEX = 66


def pool_screen_power(power_map):
    pool_size = 4
    map_size = power_map.shape[0]

    out_size = map_size // pool_size
    out = np.zeros((out_size, out_size))

    for row_index in range(out_size):
        row_num = row_index * pool_size
        for col_index in range(out_size):
            col_num = col_index * pool_size
            out[row_index, col_index] = int(np.all(power_map[row_num:row_num + pool_size, col_num:col_num + 4]))

    return out


def get_power_mask_minimap(obs):
    minimap_camera = obs.observation["minimap"][3]
    screen_power = obs.observation["screen"][3]

    screen_unit_type = obs.observation["screen"][_UNIT_TYPE]
    screen_unit = (screen_unit_type == 0).astype("int")

    reduce_screen_power = np.logical_and(screen_power, screen_unit).astype("int")
    trans_power = pool_screen_power(reduce_screen_power).reshape(-1)

    minimap_camera = minimap_camera.reshape(-1)
    minimap_camera[minimap_camera == 1] = trans_power

    return (minimap_camera == 1).astype("int")


def dialted_unit(screen_unit, size=1):
    struct = ndimage.generate_binary_structure(2, 1)
    dialted_screen_unit = ndimage.binary_dilation(screen_unit,
                                                  structure=struct, iterations=size).astype(screen_unit.dtype)
    return dialted_screen_unit


def dialted_area(area, size=1):
    struct = ndimage.generate_binary_structure(2, 1)
    dialted_area = ndimage.binary_dilation(area,
                                           structure=struct, iterations=size).astype(area.dtype)
    return dialted_area


def get_power_mask_screen(obs, size=None, show=False):
    screen_power = obs.observation["screen"][3]
    screen_unit_type = obs.observation["screen"][_UNIT_TYPE]
    screen_unit = (screen_unit_type != 0).astype("int")
    area_1 = dialted_area(1 - get_available_area(obs), size=size)
    area_2 = dialted_area(screen_unit, size=size)
    area_3 = dialted_area(1 - screen_power, size=size)

    reduce_area = np.logical_and(1 - area_1, 1 - area_2).astype("int")
    reduce_area = np.logical_and(reduce_area, 1 - area_3).astype("int")

    if show:
        imgplot = plt.imshow(reduce_area)
        plt.show()
    return reduce_area.reshape(-1)


def get_pos(pos_prob_array):
    if pos_prob_array.sum() != 0:
        pos = np.random.choice(64 * 64, size=1, p=(pos_prob_array / pos_prob_array.sum()))
    else:
        pos = 0

    x = pos % 64
    y = pos // 64
    return [x, y]


def get_available_area(obs, show=False):
    height_type = obs.observation["screen"][_HEIGHT_MAP].astype("int")
    area_1 = (height_type == np.amax(height_type)).astype("int")
    visiable_type = obs.observation["screen"][_VISIABLE_MAP].astype("int")
    area_2 = (visiable_type == np.amax(visiable_type)).astype("int")
    available_area = np.logical_and(area_1, area_2).astype("int")
    if show:
        imgplot = plt.imshow(available_area)
        plt.show()
    return available_area


def get_unit_mask_screen(obs, size=None, show=False):
    screen_unit_type = obs.observation["screen"][_UNIT_TYPE]
    screen_unit = (screen_unit_type != 0).astype("int")
    not_available_area = np.logical_or(1 - get_available_area(obs), screen_unit).astype("int")
    if size:
        not_available_area = dialted_area(not_available_area, size=size)
    reduce_area = 1 - not_available_area
    if show:
        imgplot = plt.imshow(reduce_area)
        plt.show()
    return reduce_area.reshape(-1)


def find_unit(obs, index):
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type == index and unit.build_progress == 1:
            return unit

    return None


def find_unit_on_screen(obs, index):
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type == index and unit.build_progress == 1 and unit.is_on_screen:
            return unit
    return None


def check_base_camera(game_info, obs):
    camera_world_pos = obs.raw_observation.observation.raw_data.player.camera
    camera_minimap_pos = T.world_to_minimap_pos(game_info, camera_world_pos)
    if camera_minimap_pos == C.base_camera_pos:
        return True
    return False


def get_minimap_data(timestep, verbose=False):
    obs = timestep
    map_width = 64
    relative_type_map = obs.observation["minimap"][C._M_RELATIVE_TYPE].reshape(-1, map_width, map_width) / 255
    if verbose:
        imgplot = plt.imshow(relative_type_map[0])
        plt.show()
    visiable_type_map = obs.observation["minimap"][C._M_VISIABLE_TYPE].reshape(-1, map_width, map_width) / 255
    player_id_map = obs.observation["minimap"][C._M_PLAYID_TYPE].reshape(-1, map_width, map_width) / 255
    if verbose:
        imgplot = plt.imshow(visiable_type_map[0])
        plt.show()
    if verbose:
        imgplot = plt.imshow(player_id_map[0])
        plt.show()
    map_data = np.concatenate([relative_type_map, visiable_type_map, player_id_map], axis=0)
    return map_data


def find_gas(obs, index):
    # index == 1 or 2
    gas = []
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type == C._ASSIMILATOR_TYPE_INDEX:
            # if unit is visible and on screen
            if unit.alliance == _SELF_INDEX:
                gas.append(unit)
    if len(gas) == 2:
        if gas[0].pos.x > gas[1].pos.x:
            tmp = gas[0]
            gas[0] = gas[1]
            gas[1] = tmp
        return gas[index - 1]
    elif len(gas) == 1:
        return gas[0]

    return None


def find_gas_pos(obs, index):
    # index == 1 or 2
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type == C._GAS_TYPE_INDEX:
            if unit.display_type == _VISIBLE_INDEX and unit.is_on_screen == True:
                print(unit)
    return None


def find_initial_gases(obs):
    # index == 1 or 2
    gas = []
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type == C._GAS_TYPE_INDEX:
            # if unit is visible and on screen
            if unit.display_type == _VISIBLE_INDEX and unit.is_on_screen == True:
                gas.append(unit)
    if len(gas) == 2:
        if gas[0].pos.x > gas[1].pos.x:
            tmp = gas[0]
            gas[0] = gas[1]
            gas[1] = tmp

        # print(gas[0])
        # print(gas[1])

        return gas

    return None


def is_assimilator_on_gas(obs, gas):
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type == C._ASSIMILATOR_TYPE_INDEX:
            if unit.pos.x == gas.pos.x and unit.pos.y == gas.pos.y:
                return True
    return False


def get_unit_num(obs, unit_type):
    num = 0
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type == unit_type:
            num += 1

    return num


def get_unit_num_array(obs, unit_type_list):
    num_array = np.zeros(len(unit_type_list))

    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type in unit_type_list:
            num_array[unit_type_list.index(unit.unit_type)] += 1

    return np.array(num_array)


def get_tech_action_num(obs, action_id):
    num = 0
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.orders:
            if unit.orders[0].ability_id == action_id:
                num += 1

    return num


def judge_gas_worker_too_many(obs):
    gas_1 = find_gas(obs, 1)
    gas_2 = find_gas(obs, 2)
    have_gas_1, have_gas_2 = 0, 0
    if gas_1:
        if gas_1.assigned_harvesters > gas_1.ideal_harvesters:
            have_gas_1 = 1
    if gas_2:
        if gas_2.assigned_harvesters > gas_2.ideal_harvesters:
            have_gas_2 = 1
    if have_gas_1 + have_gas_2 > 0:
        return True
    else:
        return False


def judge_gas_worker(obs, game_info):
    gas_1 = find_gas(obs, 1)
    gas_2 = find_gas(obs, 2)
    if gas_1:
        a = gas_1.assigned_harvesters
        i = gas_1.ideal_harvesters
        if a < i:
            return T.world_to_screen_pos(game_info, gas_1.pos, obs)
            # return C.gas1_pos
    if gas_2:
        a = gas_2.assigned_harvesters
        i = gas_2.ideal_harvesters
        if a < i:
            return T.world_to_screen_pos(game_info, gas_2.pos, obs)
            # return C.gas2_pos

    return None


def get_gas_probe(obs):
    # if have resources, back to base
    buff = None
    unit_set = obs.raw_observation.observation.raw_data.units

    for unit in unit_set:
        if unit.unit_type == C._PROBE_TYPE_INDEX and unit.is_on_screen == True:
            buff = unit.buff_ids
            # [274] gas
            # [271] mineal
            if buff and buff[0] == 274:
                return unit


def get_mineral_probe(obs):
    # if have resources, back to base
    buff = None
    unit_set = obs.raw_observation.observation.raw_data.units

    for unit in unit_set:
        if unit.unit_type == C._PROBE_TYPE_INDEX and unit.is_on_screen == True:
            buff = unit.buff_ids
            # [274] gas
            # [271] mineral
            if buff and buff[0] == 271:
                return unit


def get_back_pos(obs, game_info):
    # if have resources, back to base
    buff = None
    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type == C._PROBE_TYPE_INDEX and unit.is_selected == True:
            buff = unit.buff_ids
            # print('buff:', buff)

    if buff and buff[0] > 0:
        # return C.base_pos
        base = find_unit_on_screen(obs, C._NEXUS_TYPE_INDEX)
        base_pos = T.world_to_screen_pos(game_info, base.pos, obs) if base else None
        return base_pos

    base = find_unit_on_screen(obs, C._NEXUS_TYPE_INDEX)
    # number of probe for mineral and the ideal num
    if base:
        a = base.assigned_harvesters
        i = base.ideal_harvesters
        if a < i:
            mineral = find_unit_on_screen(obs, C._MINERAL_TYPE_INDEX)
            mineral_pos = T.world_to_screen_pos(game_info, mineral.pos, obs) if mineral else None
            return mineral_pos
            # return C.mineral_pos

    gas_1 = find_gas(obs, 1)
    gas_2 = find_gas(obs, 2)
    if gas_1:
        a = gas_1.assigned_harvesters
        i = gas_1.ideal_harvesters
        if a < i:
            return T.world_to_screen_pos(game_info, gas_1.pos, obs)
            # return C.gas1_pos
    if gas_2:
        a = gas_2.assigned_harvesters
        i = gas_2.ideal_harvesters
        if a < i:
            return T.world_to_screen_pos(game_info, gas_2.pos, obs)
            # return C.gas2_pos
    return T.world_to_screen_pos(game_info, base.pos, obs) if base else None
    # return C.mineral_pos


def get_production_num(obs, train_order_list):
    num_list = np.zeros(len(train_order_list))

    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.alliance == 1 and unit.orders:
            for order in unit.orders:
                if order.ability_id in train_order_list:
                    index = train_order_list.index(order.ability_id)
                    num_list[index] += 1

    return num_list


def get_best_gateway(obs):
    unit_set = obs.raw_observation.observation.raw_data.units
    best_unit = None

    for unit in unit_set:
        if unit.unit_type == C._GATEWAY_TYPE_INDEX and unit.build_progress == 1:
            if (not best_unit) or (not unit.orders) or len(best_unit.orders) > len(unit.orders):
                best_unit = unit
            if not best_unit.orders:
                return best_unit

    return best_unit

def get_production_num_and_progress(obs, train_order_list):
    num_list = np.zeros(len(train_order_list))
    progress_list = np.zeros(len(train_order_list))

    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.alliance == 1 and unit.orders:
            for order in unit.orders:
                if order.ability_id in train_order_list:
                    index = train_order_list.index(order.ability_id)
                    num_list[index] += 1
                    if order.progress > progress_list[index]:
                        progress_list[index] = int(order.progress * 100)

    return num_list, progress_list

def get_attack_num(obs, army_index_list):
    unit_set = obs.raw_observation.observation.raw_data.units
    num = 0

    for unit in unit_set:
        if unit.unit_type in army_index_list:
            if unit.orders:
                for order in unit.orders:
                    if order.ability_id == C._A_ATTACK_ATTACK_MINIMAP_S:
                        num += 1

    return num

def get_unit_num_and_progress(obs, unit_type_list):
    num_array = np.zeros(len(unit_type_list))
    progress_list = np.zeros(len(unit_type_list))

    unit_set = obs.raw_observation.observation.raw_data.units
    for unit in unit_set:
        if unit.unit_type in unit_type_list:
            index = unit_type_list.index(unit.unit_type)
            if unit.build_progress == 1.0:
                num_array[index] += 1
            elif unit.build_progress > progress_list[index]:
                progress_list[index] = int(unit.build_progress * 100)

    return num_array, progress_list

def SetPlotRC():
    # If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)


def ApplyFont(ax):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()

    text_size = 20.0

    for t in ticks:
        t.set_fontname('Times New Roman')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

def predict_state_diff_by_rule(state, action):
    simple_input = state

    # always add 2 seconds
    time = simple_input[0]

    minerals = simple_input[1]
    food_workers = simple_input[2]
    on_build_probe_num = simple_input[3]
    first_probe_progress = simple_input[4]

    pylon_build_order = simple_input[5]
    pylon_num = simple_input[6]
    first_pylon_progress = simple_input[7]

    food_cap = simple_input[8]
    food_used = simple_input[9]

    # now rule the diff of state
    diff_state = np.zeros([C._SIZE_SIMPLE_INPUT])

    # every two seconds probe progress add 15 percent
    probe_move = 15
    # every two seconds pylon progress add 11 percent
    pylon_move = 11
    probe_price = 50
    pylon_price = 100

    diff_time = +2
    diff_mineral = 0
    diff_food_workers = 0
    if first_probe_progress + probe_move >= 100:
        diff_food_workers += 1

    diff_on_build_probe_num = 0
    if action == 1 and minerals > probe_price and on_build_probe_num < 5 and food_used < food_cap:
        diff_on_build_probe_num += 1
        diff_mineral -= probe_price
    if first_probe_progress + probe_move >= 100:
        diff_on_build_probe_num -= 1

    diff_first_probe_progress = 0
    if on_build_probe_num >= 1:
        new_probe_progress = first_probe_progress + probe_move
        if new_probe_progress >= 100:
            new_probe_progress -= 100
        diff_first_probe_progress = new_probe_progress - first_probe_progress

    diff_pylon_num = 0
    if first_pylon_progress + pylon_move >= 100:
        diff_pylon_num += 1

    diff_pylon_build_order = 0
    if action == 2 and minerals > pylon_price:
        diff_pylon_build_order += 1
        diff_mineral -= pylon_price
    elif pylon_build_order > 0:
        diff_pylon_build_order -= 1

    diff_first_pylon_progress = 0
    if diff_pylon_build_order < 0 or first_pylon_progress > 0:
        new_pylon_progress = first_pylon_progress + pylon_move
        if new_pylon_progress >= 100:
            new_pylon_progress = 0
        diff_first_pylon_progress = new_pylon_progress - first_pylon_progress

    diff_food_cup = 0
    if first_pylon_progress + pylon_move >= 100:
        diff_food_cup += 8

    diff_food_used = 0
    # if action == 1 and food_used < food_cap and minerals > probe_price and on_build_probe_num < 5:
    if on_build_probe_num > 1 and first_probe_progress + probe_move > 100:
        diff_food_used += 1

    diff_state[0] = diff_time
    diff_state[1] = diff_mineral                      # use model to predict minerals
    diff_state[2] = diff_food_workers
    diff_state[3] = diff_on_build_probe_num
    diff_state[4] = diff_first_probe_progress
    diff_state[5] = diff_pylon_build_order
    diff_state[6] = diff_pylon_num
    diff_state[7] = diff_first_pylon_progress
    diff_state[8] = diff_food_cup
    diff_state[9] = diff_food_used
    return diff_state


GAME_INITIAL_SIMPLE_STATE = np.array([0, 50, 12, 0, 0, 0, 0, 0, 15, 12])


def show_prob_dist(probs, show=False, max_y=1 ,color='b', action_num=10, save=False, name="S", count=0):
    import seaborn as sns
    sns.set()
    plt.figure(figsize=(11, 6), dpi=80)
    plt.subplot(1, 1, 1)

    n_groups = action_num
    #print('probs:', probs)
    means_men = probs
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.95
    error_config = {'ecolor': '0.3'}

    if action_num == 10:
        labels = {'Do_nothing', 'Build_pylon', 'Build_gateway', 'Build_Assimilator', 'Build_CyberneticsCore', \
            'Build_probe', 'Build_zealot', 'Build_Stalker', 'Attack', 'Retreat'}
    else:
        labels = {'', '', '', '', '',}


    rects1 = plt.bar(index, means_men, bar_width,
                alpha=opacity, color=color,
                label='Probs')
    '''
    rects2 = ax.bar(index + bar_width, means_women, bar_width,
                alpha=opacity, color='r',
                yerr=std_women, error_kw=error_config,
                label='Women')'''

    plt.xlabel('Action')
    plt.ylabel('Probs')
    plt.title(name + ' Probabilty distribution')
    if action_num == 10:
        plt.xticks(index, ['NoOP', 'Pylon', 'Gateway', 'Asimitor', 'Cyber', \
            'Probe', 'Zealot', 'Stalker', 'Attack', 'Retreat'], size='small')
    else:
        plt.xticks(index, ['NoOP', 'Pylon', 'Gateway', 'Asimitor', 'Cyber', \
            'Probe', 'Zealot', 'Stalker', 'Attack', \
            'Retreat', 'AtkQ', 'RetreatQ', 'Gas', 'Atk1st', 'Atk2nd'], size='small')
    plt.legend(loc="upper right")

    #plt.xticks(range(1, len(labels)+1), labels, size='small')
    # add
    plt.ylim((0, max_y))
    plt.tight_layout()

    if save:
        plt.savefig('fig/' + name + '_' + str(count) + '.pdf')

    if show:
        plt.show()


def get_map_data(obs):
    map_width = 64
    m_height = obs.observation["minimap"][C._M_HEIGHT].reshape(-1, map_width, map_width) / 255.
    m_visible = obs.observation["minimap"][C._M_VISIBILITY].reshape(-1, map_width, map_width) / 2.
    m_camera = obs.observation["minimap"][C._M_CAMERA].reshape(-1, map_width, map_width)
    m_relative = obs.observation["minimap"][C._M_RELATIVE].reshape(-1, map_width, map_width) / 4.
    m_selected = obs.observation["minimap"][C._M_SELECTED].reshape(-1, map_width, map_width)

    s_relative = obs.observation["screen"][C._S_RELATIVE].reshape(-1, map_width, map_width) / 4.
    s_selected = obs.observation["screen"][C._S_SELECTED].reshape(-1, map_width, map_width)
    s_hitpoint = obs.observation["screen"][C._S_HITPOINT_R].reshape(-1, map_width, map_width) / 255.
    s_shield = obs.observation["screen"][C._S_SHIELD_R].reshape(-1, map_width, map_width) / 255.
    s_density = obs.observation["screen"][C._S_DENSITY_A].reshape(-1, map_width, map_width) / 255.

    map_data = np.concatenate([m_height, m_visible, m_camera, m_relative, m_selected,
                               s_relative, s_selected, s_hitpoint, s_shield, s_density], axis=0)
    return map_data


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def get_small_simple_map_data(obs, show_original=False, show_resacel=False):
    use_small_map = P.use_small_map

    map_width = 64
    small_map_width = 32

    resize_type = np.uint8
    save_type = np.float16

    m_height_origin = obs.observation["minimap"][C._M_HEIGHT]
    if show_original:
        imgplot = plt.imshow(m_height_origin)
        plt.show()
    m_height_rescaled = rescale(m_height_origin, 0.5, preserve_range=True, anti_aliasing=False)
    if show_resacel:
        imgplot = plt.imshow(m_height_rescaled)
        plt.show()   

    m_visible_origin = obs.observation["minimap"][C._M_VISIBILITY]
    if show_original:
        imgplot = plt.imshow(m_visible_origin)
        plt.show()
    m_visible_rescaled = rescale(m_visible_origin, 0.5, order=0, preserve_range=True, anti_aliasing=False).astype(resize_type)
    if show_resacel:
        imgplot = plt.imshow(m_visible_rescaled)
        plt.show()


    m_relative_origin = obs.observation["minimap"][C._M_RELATIVE]
    if show_original:
        imgplot = plt.imshow(m_relative_origin)
        plt.show()
    m_relative_rescaled = rescale(m_relative_origin, 0.5, order=0, preserve_range=True, anti_aliasing=False).astype(resize_type)
    if show_resacel:
        imgplot = plt.imshow(m_relative_rescaled)
        plt.savefig('fig/m_relative_rescaled.pdf')
        plt.show()
    
    if use_small_map:
        m_height = np.expand_dims(m_height_rescaled.reshape(-1, small_map_width, small_map_width), -1).astype(save_type) / 255.0
        m_visible = get_one_hot(m_visible_rescaled.reshape(-1, small_map_width, small_map_width), 4).astype(save_type)
        m_relative = get_one_hot(m_relative_rescaled.reshape(-1, small_map_width, small_map_width), 5).astype(save_type)
    else:
        m_height = np.expand_dims(obs.observation["minimap"][C._M_HEIGHT].reshape(-1, map_width, map_width), -1).astype(save_type) / 255.0
        m_visible = get_one_hot(obs.observation["minimap"][C._M_VISIBILITY].reshape(-1, map_width, map_width), 4).astype(save_type)
        m_relative = get_one_hot(obs.observation["minimap"][C._M_RELATIVE].reshape(-1, map_width, map_width), 5).astype(save_type)

    #do not use screen information
    #s_relative = obs.observation["screen"][C._S_RELATIVE].reshape(-1, map_width, map_width)

    out_channels = 1 + 4 + 5

    simple_map_data = np.concatenate([m_height, m_visible, m_relative], axis=3)
    #out_map_data = np.transpose(simple_map_data, [0, 2, 3, 1])

    out_data = np.squeeze(simple_map_data, axis=0)

    #print('out_data.shape:', out_data.shape)

    return out_data


def edge_state():
    feautre_idx = []

    # 0 means =, -1 means <, +1 means >
    feautre_relation = []
    
    feautre_count = []

    # edge state 1
    # probe > 20, zealot > 20
    feautre_idx.append([14,12])
    feautre_relation.append([1,1])
    feautre_count.append([15,10])

    # edge state 2
    # probe = 0, pylon > 3
    feautre_idx.append([14,9])
    feautre_relation.append([-1,1])
    feautre_count.append([15,2])

    # edge state 3
    # probe = 20, pylon = 0
    feautre_idx.append([14,9])
    feautre_relation.append([1,-1])
    feautre_count.append([15,2])

    # edge state 4
    # probe = 20, zealot = 0
    feautre_idx.append([14,12])
    feautre_relation.append([1,-1])
    feautre_count.append([15,5])

    # edge state 5
    # probe = 20, gateway = 0
    feautre_idx.append([14,8])
    feautre_relation.append([1,-1])
    feautre_count.append([15,2])

    # edge state 6
    # probe = 20, gateway = 5
    feautre_idx.append([14,8])
    feautre_relation.append([1,1])
    feautre_count.append([15,4])

    # edge state 7, for test
    # probe < 20, zealot < 20
    #feautre_idx.append([14,12])
    #feautre_relation.append([-1,-1])
    #feautre_count.append([20,20])

    feature_dict = {'idx':feautre_idx,'relation':feautre_relation,'count':feautre_count}

    return feature_dict

def calculate_state_mapping(state, feature_dict):
    feautre_idx = feature_dict['idx']
    feautre_relation = feature_dict['relation']    
    feautre_count = feature_dict['count']

    match_list = []

    for i, idx in enumerate(feautre_idx):
        relation = feautre_relation[i]
        count = feautre_count[i]
        match = True
        for j, zx in enumerate(idx):
            x = state[zx]
            y = count[j]
            r = relation[j]
            if r == 0:
                if not x == y:
                   match = False 
            elif r == -1:
                if not x < y:
                   match = False 
            elif r == 1:
                if not x > y:
                   match = False 
            else:
                raise Exception
        if match:
            match_list.append(1)
        else:
            match_list.append(0)

    return match_list

def get_simple_map_data(obs):
    map_width = 64
    save_type = np.float16

    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

    # similar as alphastar
    m_camera = get_one_hot(obs.observation["minimap"][C._M_CAMERA].reshape(-1, map_width, map_width), 2).astype(save_type)
    m_height = np.expand_dims(obs.observation["minimap"][C._M_HEIGHT].reshape(-1, map_width, map_width), -1).astype(save_type) / 255.0
    m_visible = get_one_hot(obs.observation["minimap"][C._M_VISIBILITY].reshape(-1, map_width, map_width), 4).astype(save_type)
    m_relative = get_one_hot(obs.observation["minimap"][C._M_RELATIVE].reshape(-1, map_width, map_width), 5).astype(save_type)

    #do not use screen information
    #s_relative = obs.observation["screen"][C._S_RELATIVE].reshape(-1, map_width, map_width)

    out_channels = 2 + 1 + 4 + 5

    simple_map_data = np.concatenate([m_camera, m_height, m_visible, m_relative], axis=3)
    #out_map_data = np.transpose(simple_map_data, [0, 2, 3, 1])

    out_data = np.squeeze(simple_map_data, axis=0)

    #print('out_data.shape:', out_data.shape)

    return out_data

def get_simple_state(obs):
    simple_input = np.zeros([C._SIZE_SIMPLE_INPUT])

    # minerals and object
    simple_input[0] = int(obs.raw_observation.observation.game_loop // 22.4)
    simple_input[1] = obs.raw_observation.observation.player_common.minerals
    simple_input[2] = obs.raw_observation.observation.player_common.food_workers

    # hidden information 1, probe building and progress
    production_list, progress_list = get_production_num_and_progress(obs, [C._A_TRAIN_PROBE, C._A_BUILD_PYLON_S])
    simple_input[3] = production_list[0]
    simple_input[4] = progress_list[0]

    # hidden information 2, probe order to build a pylon
    simple_input[5] = production_list[1]

    # hidden information 3, pylon building (100%) and the first one progress
    unit_production_list, unit_progress_list = get_unit_num_and_progress(obs, [C._PYLON_TYPE_INDEX])
    simple_input[6] = unit_production_list[0]
    simple_input[7] = unit_progress_list[0]

    player_common = obs.raw_observation.observation.player_common
    simple_input[8] = player_common.food_cap
    simple_input[9] = player_common.food_used

    return simple_input


def get_input(obs):
    high_input = np.zeros([C._SIZE_HIGH_NET_INPUT])
    tech_cost = np.zeros([C._SIZE_TECH_NET_INPUT])
    pop_num = np.zeros([C._SIZE_POP_NET_INPUT])

    # ###################################  high input ##########################################
    high_input[0] = C.difficulty
    # time
    high_input[1] = int(obs.raw_observation.observation.game_loop)
    # minerals
    high_input[2] = obs.raw_observation.observation.player_common.minerals
    # gas
    high_input[3] = obs.raw_observation.observation.player_common.vespene
    # mineral cost
    high_input[4] = obs.raw_observation.observation.score.score_details.spent_minerals
    # gas cost
    high_input[5] = obs.raw_observation.observation.score.score_details.spent_vespene
    # others
    player_common = obs.raw_observation.observation.player_common
    high_input[6] = player_common.food_cap
    high_input[7] = player_common.food_used
    high_input[8] = player_common.food_army
    high_input[9] = player_common.food_workers
    high_input[10] = player_common.army_count
    # num of probe, zealot, stalker, pylon, assimilator, gateway, cyber
    index_list = [C._PROBE_TYPE_INDEX, C._ZEALOT_TYPE_INDEX, C._STALKER_TYPE_INDEX,
                  C._PYLON_TYPE_INDEX, C._ASSIMILATOR_TYPE_INDEX, C._GATEWAY_TYPE_INDEX, C._CYBER_TYPE_INDEX]
    high_input[11:18] = get_unit_num_array(obs, index_list)

    high_input[18] = get_attack_num(obs, [C._ZEALOT_TYPE_INDEX, C._STALKER_TYPE_INDEX])

    # print('high_input:', high_input)

    # ##################################  tech cost ###############################################
    # cost of pylon vespene gateway cyber
    tech_cost[:4] = [100, 75, 150, 150]
    tech_cost[4] = player_common.food_cap - player_common.food_used
    tech_cost[5] = get_tech_action_num(obs, C._A_BUILD_PYLON_S)
    tech_cost[6] = get_tech_action_num(obs, C._A_BUILD_ASSIMILATOR_S)
    tech_cost[7] = get_tech_action_num(obs, C._A_BUILD_GATEWAY_S)
    tech_cost[8] = get_tech_action_num(obs, C._A_BUILD_CYBER_S)

    # #################################  pop num  #################################################
    base = find_unit(obs, C._NEXUS_TYPE_INDEX)
    # number of probe for mineral and the ideal num
    if base:
        pop_num[0] = base.assigned_harvesters
        pop_num[1] = base.ideal_harvesters

    gas_1 = find_gas(obs, 1)
    gas_2 = find_gas(obs, 2)
    have_gas_1, have_gas_2 = 0, 0
    if gas_1:
        pop_num[2] = gas_1.assigned_harvesters
        pop_num[3] = gas_1.ideal_harvesters
        have_gas_1 = 1
    if gas_2:
        pop_num[4] = gas_2.assigned_harvesters
        pop_num[5] = gas_2.ideal_harvesters
        have_gas_2 = 1

    # all the num of workers
    pop_num[6] = player_common.food_army / max(player_common.food_workers, 1)

    pop_num[7] = have_gas_1
    pop_num[8] = have_gas_2

    # num of the training probe, zealot, stalker
    production_list = get_production_num(obs, [C._A_TRAIN_PROBE, C._A_TRAIN_ZEALOT, C._A_TRAIN_STALKER])
    pop_num[9] = production_list[0]
    pop_num[10] = production_list[1]
    pop_num[11] = production_list[2]

    # pop_num[[12, 13, 14]] = [50, 100, 125]

    return high_input, tech_cost, pop_num
