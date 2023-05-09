import string
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import minmax_scale
# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import sys
sys.path.append('..')
from src.model import Model


def delete_punctuation(text):
    exclude = string.punctuation + "«»\n-"
    table = str.maketrans(exclude, " " * len(exclude))
    return text.translate(table).lower()


def format_str(text):
    words = [word for word in delete_punctuation(text).split()]
    return ' '.join(words)


def prepare_ratings():
    ratings = pd.read_csv('../../old_ratings.csv')
    new_ratings = pd.read_csv('../../ratings.csv', header=None, names=['user_id', 'book_id', 'rating'])
    ratings = pd.concat([ratings, new_ratings], ignore_index=True)

    ratings['book_id'] -= 1
    ratings['user_id'] -= 1
    new_ratings['book_id'] -= 1
    new_ratings['user_id'] -= 1

    ratings = sp.coo_matrix((np.array(ratings.rating.values, dtype=np.int64),
                             (np.array(ratings.user_id.values, dtype=np.int64),
                              np.array(ratings.book_id.values, dtype=np.int64))))
    return ratings


def prepare_books():
    books = pd.read_csv('../../translated_books.csv')
    gid = books['goodreads_book_id']
    books.drop(labels=['ratings_count', 'work_ratings_count', 'work_text_reviews_count',
                       'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
                       'average_rating', 'original_title', 'автор', 'prep', 'sum', 'inv_sum',
                       'work_id', 'goodreads_book_id', 'book_id', 'isbn', 'isbn13', 'title',
                       'image_url', 'small_image_url', 'best_book_id',
                       'books_count', 'language_code'], axis=1, inplace=True)

    books.fillna(books.original_publication_year.mean(), inplace=True)

    authors = {}

    for i, line in enumerate(books.authors.unique()):
        temp = list(map(lambda x: x.strip(), line.split(',')))
        for author in temp:
            if not authors.get(author):
                authors[author] = [0] * len(books)
            authors[author][i] = 1
    pref = set()
    for key in authors.keys():
        s = sum(authors[key])
        if s > 1:
            pref.add(key)
    pref.add('Others')

    authors['Others'] = [0] * len(books)
    d = authors.copy()
    for author in d.keys():
        if author not in pref:
            for i in range(len(books)):
                if d[author][i]:
                    authors['Others'] = 1
            del authors[author]

    authors = pd.DataFrame.from_dict(authors)
    books.drop(labels=['authors'], axis=1, inplace=True)

    X = books.values
    X = minmax_scale(X)
    books = pd.DataFrame(X, columns=books.columns)
    books = pd.concat((books, authors), axis=1)
    book_features = sp.csr_matrix(books)

    # tags = pd.read_csv('../tags.csv')
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

    return book_features


def prepare_preds(model: Model):
    new_ratings = pd.read_csv('ratings.csv', header=None, names=['user_id', 'book_id', 'rating'])

    new_ratings['book_id'] -= 1
    new_ratings['user_id'] -= 1
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

    preds = pd.DataFrame(data=zip(pred_u, pred_b), index=None, columns=['user_id', 'book_id'])
    preds['score'] = model.predict(preds.user_id.values, preds.book_id.values,
                                   item_features=model.book_features)
    preds.sort_values(by=['user_id', 'score'], ascending=[True, False], inplace=True)

    predictions = pd.DataFrame(columns=['user_id', 'book_id', 'score'])
    for key in u.keys():
        predictions = pd.concat((predictions, preds.loc[preds['user_id'] == key].head(50)))

    predictions.drop(labels=['score'], axis=1, inplace=True)
    predictions.to_csv('predictions.txt', index=False)
