from abc import ABC, abstractmethod
from typing import Dict, List


class IFacialMocapPoseConverter(ABC):
    @abstractmethod
    def convert(self, ifacialmocap_pose: Dict[str, float], isPerfectsync: bool) -> List[float]:
        pass

    @abstractmethod
    def init_pose_converter_panel(self, parent):
        pass