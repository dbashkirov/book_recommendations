
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from lightfm import LightFM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from multiprocessing import cpu_count
from sklearn.preprocessing import minmax_scale

while 1:
    ratings = pd.read_csv('old_ratings.csv')
    new_ratings = pd.read_csv('ratings.csv', header=None, names=['user_id', 'book_id', 'rating'])
    ratings = pd.concat([ratings, new_ratings], ignore_index=True)

    ratings['book_id'] -= 1
    ratings['user_id'] -= 1
    new_ratings['book_id'] -= 1
    new_ratings['user_id'] -= 1

    ratings = sp.coo_matrix((np.array(ratings.rating.values, dtype=np.int64),
                             (np.array(ratings.user_id.values, dtype=np.int64),
                              np.array(ratings.book_id.values, dtype=np.int64))))

    books = pd.read_csv('translated_books.csv')
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

    model = LightFM(no_components=50, loss='warp', random_state=42)
    model.fit(ratings, item_features=book_features, num_threads=cpu_count(), epochs=20)

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
        k = 0
        for i in range(10000):
            if k == 50:
                break
            if i not in u[key]:
                pred_u.append(int(key))
                pred_b.append(i)
                k += 1

    preds = pd.DataFrame(data=zip(pred_u, pred_b), index=None, columns=['user_id', 'book_id'])
    preds['score'] = model.predict(preds.user_id.values, preds.book_id.values,
                                   item_features=book_features)

    preds.sort_values(by=['user_id', 'score'], ascending=[True, False], inplace=True)
    preds.drop(labels=['score'], axis=1, inplace=True)
    preds.to_csv('predictions.csv', index=False)
    print('Done')
