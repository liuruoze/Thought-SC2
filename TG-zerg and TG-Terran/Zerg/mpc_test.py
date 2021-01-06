from strategy.protoss_agent import Protoss
from strategy.protoss_agent import ProtossAction
from strategy.agent import Dummy
from mpc.shooting import MPC
from mapping_env import SimulatePlatform

import unit.protoss_unit as P
import unit.terran_unit as T


def net_test():
    red_agent = Protoss()
    max_actions = ProtossAction.All.value
    red_agent.set_mpc(MPC(max_actions))
    blue_agent = Dummy()
    blue_agent.add_unit(T.Marine(), 5)
    blue_agent.add_building(T.Commandcenter(), 1)
    blue_agent.add_building(T.Supplydepot(), 3)
    blue_agent.add_building(T.Barracks(), 1)

    env = SimulatePlatform(red_agent=red_agent, blue_agent=blue_agent,
                           distance=5, max_steps=100)
    env.init()
    # env.simulate()
    red_agent.play_with_mpc(verbose=False)

if __name__ == "__main__":
    net_test()
