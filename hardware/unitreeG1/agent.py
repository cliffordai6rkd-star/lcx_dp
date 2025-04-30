from typing import Text, Mapping, Any

from hardware.unitreeG1.dex3 import Dex3
from hardware.base.robot import Robot
import glog as log

class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any]):
        self._dex3_l = Dex3()
        self._dex3_r = Dex3(isLeft=False)

    def print_state(self):
        log.info(f"Has realtime kernel: {libfranka.has_realtime_kernel()}")
        # self._dex3_l.print_state()
        # self._dex3_r.print_state()

    