from lightfm import LightFM
from multiprocessing import cpu_count


class Model:
    def __init__(self, no_components: int = 50):
        self.no_components = no_components
        self.model = LightFM(no_components=self.no_components, loss='warp', random_state=42)

    def fit(self, ratings, book_features, n_epochs=100):
        self.n_epochs = n_epochs
        self.ratings = ratings
        self.book_features = book_features
        self.model.fit(ratings, item_features=book_features, num_threads=cpu_count(), epochs=self.n_epochs)

    def predict(self):
        pass

    def infer(self):
        pass
