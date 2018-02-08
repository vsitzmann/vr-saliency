import os

# The path to the python exectuable
EXEC_HOME = os.path.dirname(__file__)
# The project root directory
PROJECT_ROOT = os.path.join(EXEC_HOME, os.pardir)
# The user home directory
USER_HOME = os.path.expanduser("~")

IMG_PATH = os.path.join(PROJECT_ROOT, 'data', 'img_files')
DATASET_PATH_VR = os.path.join(PROJECT_ROOT, 'data', 'vr')
DATASET_PATH_SITTING = os.path.join(PROJECT_ROOT, 'data', 'seated_vr')
DATASET_PATH_BROWSER = os.path.join(PROJECT_ROOT, 'data', 'browser')
