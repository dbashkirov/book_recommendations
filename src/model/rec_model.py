from lightfm import LightFM
from multiprocessing import cpu_count
import pickle


class Model:
    def __init__(self, no_components: int = 50):
        self.no_components = no_components
        self.model = LightFM(
            no_components=self.no_components, loss="warp", random_state=42
        )

    def fit(self, ratings, book_features, n_epochs=10):
        """
        function for fitting and saving fitted model
        Parameters
        ----------
        ratings
        book_features
        n_epochs

        Returns
        -------

        """
        self.model.fit(
            ratings,
            item_features=book_features,
            num_threads=cpu_count(),
            epochs=n_epochs,
            verbose=True,
        )
        self._save_model("models/model.pkl")

    def predict(self, user_id, item_id, item_features=None):
        """
        function for predicting scores for paires of (user_id, item_id)
        Parameters
        ----------
        user_id
        item_id
        item_features

        Returns
        -------
        prediction for pairs of (user_id, item_id)
        """
        self._load_model("models/model.pkl")
        return self.model.predict(user_id, item_id, item_features=item_features)

    def _save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_model(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
