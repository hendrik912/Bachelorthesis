import enum

class TableType(enum.Enum):
    # Enum for the different types of latex tables for the classifiers
    BY_SAMPLE_LEN = "sample_length"
    BY_DIVISION = "division"
    BY_RATIO = "ratio"
