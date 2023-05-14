import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import minmax_scale
import click

# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


@click.command()
@click.argument(
    "book_path", type=click.Path(), default="data/processed/translated_books.csv"
)
@click.argument("tags_path", type=click.Path(), default="")
def prepare_books(
    book_path: str = "data/processed/translated_books.csv", tags_path: str = ""
):
    """
    Function to prepare book data for fitting and prediction
    Parameters
    ----------
    book_path
    tags_path

    Returns
    -------

    """
    books = pd.read_csv(book_path)
    books.drop(
        labels=[
            "ratings_count",
            "work_ratings_count",
            "work_text_reviews_count",
            "ratings_1",
            "ratings_2",
            "ratings_3",
            "ratings_4",
            "ratings_5",
            "average_rating",
            "original_title",
            "автор",
            "prep",
            "sum",
            "inv_sum",
            "work_id",
            "goodreads_book_id",
            "book_id",
            "isbn",
            "isbn13",
            "title",
            "image_url",
            "small_image_url",
            "best_book_id",
            "books_count",
            "language_code",
        ],
        axis=1,
        inplace=True,
    )

    books.fillna(books.original_publication_year.mean(), inplace=True)

    authors = {}

    for i, line in enumerate(books.authors.unique()):
        temp = list(map(lambda x: x.strip(), line.split(",")))
        for author in temp:
            if not authors.get(author):
                authors[author] = [0] * len(books)
            authors[author][i] = 1
    pref = set()
    for key in authors.keys():
        s = sum(authors[key])
        if s > 1:
            pref.add(key)
    pref.add("Others")

    authors["Others"] = [0] * len(books)
    d = authors.copy()
    for author in d.keys():
        if author not in pref:
            for i in range(len(books)):
                if d[author][i]:
                    authors["Others"] = 1
            del authors[author]

    authors = pd.DataFrame.from_dict(authors)
    books.drop(labels=["authors"], axis=1, inplace=True)

    X = books.values
    X = minmax_scale(X)
    books = pd.DataFrame(X, columns=books.columns)
    books = pd.concat((books, authors), axis=1)
    book_features = sp.csr_matrix(books)

    # tags = pd.read_csv(tags_path)
    #
    # t = {}
    # for line in tags.values:
    #     if type(line[1]) == str:
    #         t[line[0]] = format_str(line[1])
    #
    # book_tags = pd.read_csv('../book_tags.csv')
    #
    # g = {}
    # book_tags.head(50)
    # for i, line in enumerate(book_tags.values):
    #     if line[0] in g:
    #         if line[1] in t:
    #             if line[2] >= 500:
    #                 g[line[0]] += ' ' + t[line[1]]
    #             else:
    #                 del t[line[1]]
    #     else:
    #         if line[1] in t:
    #             if line[2] >= 500:
    #                 g[line[0]] = t[line[1]]
    #
    # t = []
    # for i in gid:
    #     if i in g:
    #         t.append(g[i])
    #     else:
    #         t.append('')
    #
    # vectorizer = CountVectorizer()
    # vectorized_tags = vectorizer.fit_transform(t)
    # transformer = TfidfTransformer()
    # transformed_tags = transformer.fit_transform(vectorized_tags)
    #
    # book_features = sp.hstack((book_features, transformed_tags))

    click.echo("Book features prepared")
    sp.save_npz("models/book_features.npz", book_features)
    click.echo("Book features saved at")


if __name__ == "__main__":
    prepare_books()
