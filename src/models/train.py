import sys
sys.path.append('../utils')

from utils import prepare_ratings, prepare_books
from ..model import Model

ratings = prepare_ratings()
book_features = prepare_books()

model = Model()
model.fit(ratings, book_features)
