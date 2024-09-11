import enum

# Enum for the type of the latex-table for the survey
class TableType(enum.Enum):
   SUBJECT_WISE = "subject_wise"
   IMAGE_WISE = "image_wise"
   ACROSS_ALL_SUBJ = "across_all_subj"
