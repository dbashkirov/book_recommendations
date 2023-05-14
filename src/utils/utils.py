import string
import numpy as np
import pandas as pd
import scipy.sparse as sp


def delete_punctuation(text):
    exclude = string.punctuation + "«»\n-"
    table = str.maketrans(exclude, " " * len(exclude))
    return text.translate(table).lower()


def format_str(text):
    words = [word for word in delete_punctuation(text).split()]
    return " ".join(words)


def prepare_ratings(old_ratings_path: str, new_ratings_path: str):
    ratings = pd.read_csv(old_ratings_path)
    new_ratings = pd.read_csv(
        new_ratings_path, header=None, names=["user_id", "book_id", "rating"]
    )
    ratings = pd.concat([ratings, new_ratings], ignore_index=True)

    ratings["book_id"] -= 1
    ratings["user_id"] -= 1
    new_ratings["book_id"] -= 1
    new_ratings["user_id"] -= 1

    ratings = sp.coo_matrix(
        (
            np.array(ratings.rating.values, dtype=np.int64),
            (
                np.array(ratings.user_id.values, dtype=np.int64),
                np.array(ratings.book_id.values, dtype=np.int64),
            ),
        )
    )
    return ratings


def prepare_preds(
    model, new_ratings_path: str, book_features_path: str, output_path: str
):
    new_ratings = pd.read_csv(
        new_ratings_path, header=None, names=["user_id", "book_id", "rating"]
    )

    new_ratings["book_id"] -= 1
    new_ratings["user_id"] -= 1
    u = {}
    for line in new_ratings.values:
        if u.get(line[0]):
            u[line[0]].add(line[1])
        else:
            u[line[0]] = set()
            u[line[0]].add(line[1])

    pred_u = []
    pred_b = []
    for key in u.keys():
        for i in range(10000):
            if i not in u[key]:
                pred_u.append(int(key))
                pred_b.append(i)

    preds = pd.DataFrame(
        data=zip(pred_u, pred_b), index=None, columns=["user_id", "book_id"]
    )
    book_features = sp.load_npz(book_features_path)
    preds["score"] = model.predict(
        preds.user_id.values, preds.book_id.values, item_features=book_features
    )
    preds.sort_values(by=["user_id", "score"], ascending=[True, False], inplace=True)

    predictions = pd.DataFrame(columns=["user_id", "book_id", "score"])
    for key in u.keys():
        predictions = pd.concat(
            (predictions, preds.loc[preds["user_id"] == key].head(50))
        )

    predictions.drop(labels=["score"], axis=1, inplace=True)
    predictions.to_csv(output_path, index=False)
