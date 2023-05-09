from utils.utils import prepare_ratings, prepare_books
from model.model import Model

ratings = prepare_ratings()
book_features = prepare_books()

model = Model()
model.fit(ratings, book_features)
