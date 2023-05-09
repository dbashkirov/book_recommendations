import sys
sys.path.append('../../')

from utils import prepare_preds
from src import Model

model = Model()

prepare_preds(model)
