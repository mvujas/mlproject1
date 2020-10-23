# Representation of given classes as it should be in submission file.
#   These fields may be deprecated as all code uses IO helper functions
#   provided with the project.
CLASS_B = -1
CLASS_S = 1

# S is considered as a relevant class which is important information when calculating
#   recall and F_beta score of a model
CLASS_TO_BE_RETRIEVED = CLASS_S

# Maps class name to corresponding numeric value
CLASS_MAPPING = {
    'b': CLASS_B,
    's': CLASS_S
}