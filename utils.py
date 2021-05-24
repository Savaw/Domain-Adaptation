from dataclasses import dataclass
from enum import Enum  

@dataclass
class HP:
    lr:float = 0.0005
    gamma:float = 0.99


class DAModels(Enum):
    DANN = "DANN"
    MCD = "MCD"
    CDAN = "CDAN"
    # SOURCE = "SOURCE"
    MMD = "MMD"
    CORAL = "CORAL"
    SYMNET = "SYMNET"
