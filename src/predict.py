import click
from utils import prepare_preds
from model import Model


@click.command()
@click.argument("new_ratings_path", type=click.Path(), default="data/raw/ratings.csv")
@click.argument("book_features_path", type=click.Path(), default="models/book_features.npz")
@click.argument("output_path", type=click.Path(), default="data/predictions.csv")
def predict(
    new_ratings_path: str = "data/raw/ratings.csv",
    book_features_path: str = "models/book_features.npz",
    output_path: str = "data/predictions.csv",
):
    model = Model()
    click.echo("Model initialized")

    prepare_preds(model, new_ratings_path, book_features_path, output_path)


if __name__ == "__main__":
    predict()
