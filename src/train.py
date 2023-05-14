import click
from utils import prepare_ratings
from model import Model
import scipy.sparse as sp


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
    ratings = prepare_ratings(old_ratings_path, new_ratings_path)
    click.echo("Ratings prepared")

    book_features = sp.load_npz(book_features_path)
    click.echo("Book features loaded")

    model = Model()
    click.echo("Model initialized")

    model.fit(ratings, book_features, n_epochs=n_epochs)
    click.echo("Model fitted")


if __name__ == "__main__":
    train()
