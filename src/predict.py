import sys
sys.path.append('../')

from src.utils import prepare_preds
from src.model import Model

model = Model()

prepare_preds(model)
