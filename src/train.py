import click
from utils import prepare_ratings
from model import Model
import scipy.sparse as sp
import mlflow
from dotenv import load_dotenv
import os
from sklearn.metrics import ndcg_score

load_dotenv()

remote_server_uri = "http://62.217.183.172:5000"
mlflow.set_tracking_uri(remote_server_uri)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "bookrec"


@click.command()
@click.argument(
    "book_features_path", type=click.Path(), default="models/book_features.npz"
)
@click.argument(
    "old_ratings_path", type=click.Path(), default="data/raw/old_ratings.csv"
)
@click.argument("new_ratings_path", type=click.Path(), default="data/raw/ratings.csv")
@click.argument("n_epochs", type=click.INT, default=10)
def train(
    book_features_path: str = "models/book_features.npz",
    old_ratings_path: str = "data/raw/old_ratings.csv",
    new_ratings_path: str = "data/ratings.csv",
    n_epochs: int = 10,
):
    with mlflow.start_run():
        mlflow.get_artifact_uri()
        ratings, df = prepare_ratings(old_ratings_path, new_ratings_path)
        click.echo("Ratings prepared")

        book_features = sp.load_npz(book_features_path)
        click.echo("Book features loaded")

        no_components = 50
        model = Model(no_components)
        click.echo("Model initialized")

        model.fit(ratings, book_features, n_epochs=n_epochs)
        click.echo("Model fitted")

        y_pred = model.predict(df.user_id.values, df.book_id.values, book_features)
        score = ndcg_score([df.rating], [y_pred])
        click.echo(f"Train score: {score:.2f}")

        # signature = infer_signature(ratings, y_pred)

        mlflow.log_param("no_components", no_components)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_metric("ndcg_score", score)
        mlflow.log_artifact("models/model.pkl")


if __name__ == "__main__":
    train()
