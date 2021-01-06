import lib.config as C
import contrib.utils as U
from lib import transform_pos as T
import numpy as np


def check_params(agent, action, unit_type, args, action_type):
    valid = False
    if action_type == 0:  # select
        if action in agent.obs.observation["available_actions"]:
            valid = True
            if None in args:
                valid = False
    elif action_type == 1:  # action
        if action == C._NO_OP or action == C._MOVE_CAMERA:
            valid = True
        # type and action is correct, right
        elif agent.on_select == unit_type and action in agent.obs.observation["available_actions"]:
            valid = True
            if None in args:
                valid = False
    return valid


def control_step(agent):
    agent.select(C._SELECT_POINT, C._GATEWAY_GROUP_INDEX, [C._DBL_CLICK, selectGateway(agent)])
    agent.safe_action(C._CONTROL_GROUP, C._GATEWAY_GROUP_INDEX, [C._SET_GROUP, C._GATEWAY_GROUP_ID])


def attack_step(agent, pos_index=None):
    # select army and attack, first main-mineral then sub-mineral
    agent.select(C._SELECT_ARMY, C._ARMY_INDEX, [[0]])
    agent.safe_action(C._CONTROL_GROUP, C._ARMY_INDEX, [C._SET_GROUP, C._ARMY_GROUP_ID])
    # if agent.obs.raw_observation.observation.player_common.army_count > 30:
    if pos_index is None or pos_index == 0:
        for pos in C.attack_pos_queue:
        # agent.safe_action(C._ATTACH_M, C._ARMY_INDEX, [C._QUEUED, C.my_sub_pos])
            agent.safe_action(C._ATTACH_M, C._ARMY_INDEX, [C._QUEUED, pos])


def retreat_step(agent):
    # select army and assemble to our sub-mineral location
    agent.select(C._SELECT_ARMY, C._ARMY_INDEX, [[0]])
    agent.safe_action(C._CONTROL_GROUP, C._ARMY_INDEX, [C._SET_GROUP, C._ARMY_GROUP_ID])
    agent.safe_action(C._MOVE_M, C._ARMY_INDEX, [C._QUEUED, C.my_sub_pos])


def move_worker(agent, gas_pos, pos=None):
    # get a probe to gas
    agent.select(C._SELECT_POINT, C._DRONE_TYPE_INDEX, [C._CLICK, pos if pos else select_unit(agent)])
    # first back to base, then go to change the target
    # self.safe_action(C._SMART_SCREEN, C._PROBE_TYPE_INDEX, [C._QUEUED, C.base_pos])
    agent.safe_action(C._HARVEST_S, C._DRONE_TYPE_INDEX, [C._NOT_QUEUED, gas_pos])

def gather_resource(agent, resource='gas'):
    camera_on_base = U.check_base_camera(agent.env.game_info, agent.obs)

    if not camera_on_base:
        return 
    if resource == 'gas':
        gas_pos = U.judge_gas_worker(agent.obs, agent.env.game_info)
        drone = U.get_mineral_drone(agent.obs)
        drone_pos = T.world_to_screen_pos(agent.env.game_info, drone.pos, agent.obs) if drone else None
        if gas_pos and drone_pos:
            agent.select(C._SELECT_POINT, C._DRONE_TYPE_INDEX, [C._CLICK, drone_pos])
            agent.safe_action(C._HARVEST_S, C._DRONE_TYPE_INDEX, [C._NOT_QUEUED, gas_pos])
    if resource == 'mineral':
        mineral_pos = U.find_mineral(agent.obs)
        mineral_pos = T.world_to_screen_pos(agent.env.game_info, mineral_pos, agent.obs)
        drone = U.get_gas_drone(agent.obs)
        drone_pos = T.world_to_screen_pos(agent.env.game_info, drone.pos, agent.obs) if drone else None
        if mineral_pos and drone_pos:
            agent.select(C._SELECT_POINT, C._DRONE_TYPE_INDEX, [C._CLICK, drone_pos])
            agent.safe_action(C._HARVEST_S, C._DRONE_TYPE_INDEX, [C._NOT_QUEUED, mineral_pos])
    
    return

def mineral_worker(agent):
    camera_on_base = U.check_base_camera(agent.env.game_info, agent.obs)
    # print(camera_on_base)

    if not camera_on_base:
        return

    # gas_pos = U.judge_gas_worker(agent.obs, agent.env.game_info)
    # if gas_pos:
    #     drone = U.get_mineral_drone(agent.obs)
    #     drone_pos = T.world_to_screen_pos(agent.env.game_info, drone.pos, agent.obs) if drone else None
    #     move_worker(agent, gas_pos, drone_pos)

    # if U.judge_gas_worker_too_many(agent.obs):
    #     drone = U.get_gas_drone(agent.obs)
    #     drone_pos = T.world_to_screen_pos(agent.env.game_info, drone.pos, agent.obs) if drone else None

    #     mineral = U.find_unit_on_screen(agent.obs, C._MINERAL_TYPE_INDEX)
    #     mineral_pos = T.world_to_screen_pos(agent.env.game_info, mineral.pos, agent.obs) if mineral else None
    #     move_worker(agent, mineral_pos, drone_pos)

        # move_worker(agent, C.mineral_pos, probe_pos)
    # else:
        # train_worker(agent, C.base_pos, C._TRAIN_PROBE)
    base = U.find_unit_on_screen(agent.obs, C._HATCHERY_TYPE_INDEX)
    base_pos = T.world_to_screen_pos(agent.env.game_info, base.pos, agent.obs) if base else None
    train_worker(agent, base_pos, C._TRAIN_DRONE)


def train_worker(agent, building_pos, train_action, click=C._CLICK):
    # select all larvas and train
    agent.select(C._SELECT_POINT, C._HATCHERY_TYPE_INDEX, [click, building_pos])
    agent.select(C._SELECT_LARVA, C._LARVA_TYPE_INDEX, [])
    agent.safe_action(train_action, C._LARVA_TYPE_INDEX, [C._NOT_QUEUED])


def build_by_idle_worker(agent, build_action, build_pos):
    if C._SELECT_WORKER in agent.obs.observation["available_actions"]:
        agent.select(C._SELECT_WORKER, C._DRONE_TYPE_INDEX, [[0]])
    else:
        agent.select(C._SELECT_POINT, C._DRONE_TYPE_INDEX, [C._CLICK, select_unit(agent)])
    agent.safe_action(build_action, C._DRONE_TYPE_INDEX, [C._NOT_QUEUED, build_pos])
    # agent.safe_action(C._SMART_SCREEN, C._DRONE_TYPE_INDEX, [C._QUEUED, U.get_back_pos(agent.obs, agent.env.game_info)])


def train_army(agent, train_action):
    base = U.find_unit_on_screen(agent.obs, C._HATCHERY_TYPE_INDEX)
    camera_on_base = U.check_base_camera(agent.env.game_info, agent.obs)

    if (not camera_on_base) or (base is None):
        return

    base_pos = T.world_to_screen_pos(agent.env.game_info, base.pos, agent.obs)
    agent.select(C._SELECT_POINT, C._HATCHERY_TYPE_INDEX, [C._DBL_CLICK, base_pos])
    if train_action != C._TRAIN_QUEEN:
        agent.select(C._SELECT_LARVA, C._LARVA_TYPE_INDEX, [])  
        agent.safe_action(train_action, C._LARVA_TYPE_INDEX, [C._NOT_QUEUED])
    else:
        agent.safe_action(C._TRAIN_QUEEN, C._HATCHERY_TYPE_INDEX, [C._NOT_QUEUED])


def set_source(agent):
    agent.safe_action(C._NO_OP, 0, [])
    base = U.find_unit_on_screen(agent.obs, C._NEXUS_TYPE_INDEX)
    base_pos = T.world_to_screen_pos(agent.env.game_info, base.pos, agent.obs)

    agent.select(C._SELECT_POINT, C._NEXUS_TYPE_INDEX, [C._CLICK, base_pos])
    agent.safe_action(C._CONTROL_GROUP, C._NEXUS_TYPE_INDEX, [C._SET_GROUP, C._BASE_GROUP_ID])


def reset_select(agent):
    agent.select(C._CONTROL_GROUP, C._NEXUS_TYPE_INDEX, [C._RECALL_GROUP, C._BASE_GROUP_ID])


def select_unit(agent, unit_type=C._DRONE_TYPE_INDEX):
    # random select a unit
    unit_type_map = agent.obs.observation["screen"][C._S_UNIT_TYPE]
    pos_y, pos_x = (unit_type_map == unit_type).nonzero()

    num = len(pos_y)
    if num > 0:
        rand = np.random.choice(num, size=1)
        pos = [pos_x[rand], pos_y[rand]]
        return pos
    return None

def inject_larva(agent):
    base = U.find_unit_on_screen(agent.obs, C._HATCHERY_TYPE_INDEX)
    queen_pos = select_unit(agent, C._QUEEN_TYPE_INDEX)

    camera_on_base = U.check_base_camera(agent.env.game_info, agent.obs)
    if not camera_on_base or queen_pos is None or base is None:
        return
    base_pos = T.world_to_screen_pos(agent.env.game_info, base.pos, agent.obs)    
    agent.select(C._SELECT_POINT, C._QUEEN_TYPE_INDEX, [C._CLICK, queen_pos])
    agent.safe_action(C._E_INJECT_LARVA, C._QUEEN_TYPE_INDEX, [C._NOT_QUEUED, base_pos])
    

def selectGateway(agent):
    # random select a Gateway
    gateway = U.get_best_gateway(agent.obs)
    pos = T.world_to_screen_pos(agent.env.game_info, gateway.pos, agent.obs) if gateway else None
    return pos

def set_rally_point(agent, pos):
    base = U.find_unit_on_screen(agent.obs, C._HATCHERY_TYPE_INDEX)
    base_pos = T.world_to_screen_pos(agent.env.game_info, base.pos, agent.obs)
    agent.select(C._SELECT_POINT, C._HATCHERY_TYPE_INDEX, [C._DBL_CLICK, base_pos])
    agent.safe_action(C._RALLY_UNITS_M, C._HATCHERY_TYPE_INDEX, [C._NOT_QUEUED, C.rally_pos])
