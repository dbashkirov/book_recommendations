from utils import prepare_ratings, prepare_books, prepare_preds
from src.model import Model

ratings = prepare_ratings()
book_features = prepare_books()

model = Model()
model.fit(ratings, book_features)

prepare_preds(model)
