from utils import prepare_ratings
from model import Model
import scipy.sparse as sp


ratings = prepare_ratings()
print("Ratings prepared")

book_features = sp.load_npz("models/book_features.npz")
print("Book features loaded")

model = Model()
print("Model initialized")

model.fit(ratings, book_features)
