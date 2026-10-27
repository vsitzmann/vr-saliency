import os

# The path to the python exectuable
EXEC_HOME = os.path.dirname(__file__)
# The user home directory
USER_HOME = os.path.expanduser("~")

DATASET_PATH = os.path.join(EXEC_HOME, 'data', '60 Hz')
DATASET_PATH_60hz = os.path.join(EXEC_HOME, 'new_data')
DATASET_PATH_BROWSER = os.path.join(EXEC_HOME, 'standing_data')
# DATASET_PATH = '/media/vsitzmann/New Volume/Programming/eyetrackingAnalysis/data/'
EXPERIMENT_PATH = os.path.join(EXEC_HOME, 'new_data')
PREDICTION_SOURCE_PATH = '/media/vsitzmann/New Volume/Programming/eyetrackingAnalysis/to_predict/'