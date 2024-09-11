from enum import Enum

class MetricType(Enum):
    # Enum for the different types of metrics used (or experimented with)
    JSD = "JSD"
    KLD = "KLD"
    EUC = "Euc"
    IS = "IS"
    FIS = "FIS"
    SWD = "SWD"
