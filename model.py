#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import string
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from lightfm import LightFM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from multiprocessing import cpu_count
from sklearn.preprocessing import minmax_scale
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer 


# In[74]:


ratings = pd.read_csv('old_ratings.csv')
new_ratings = pd.read_csv('ratings.csv', header=None, names=['user_id', 'book_id', 'rating'])
ratings = pd.concat([ratings, new_ratings], ignore_index=True)


# In[47]:


ratings['book_id'] -= 1
ratings['user_id'] -= 1
new_ratings['book_id'] -= 1
new_ratings['user_id'] -= 1


# In[48]:


ratings = sp.coo_matrix((np.array(ratings.rating.values, dtype=np.int64),
                         (np.array(ratings.user_id.values, dtype=np.int64),
                          np.array(ratings.book_id.values, dtype=np.int64))))


# In[50]:


books = pd.read_csv('translated_books.csv')
gid = books['goodreads_book_id']
books.drop(labels=['ratings_count', 'work_ratings_count', 'work_text_reviews_count', 
                   'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5', 
                   'average_rating', 'original_title', 'автор', 'prep', 'sum', 'inv_sum',
                   'work_id', 'goodreads_book_id', 'book_id', 'isbn', 'isbn13', 'title', 
                   'image_url', 'small_image_url', 'best_book_id', 
                   'books_count', 'language_code'], axis=1, inplace=True)


# In[51]:


books.fillna(books.original_publication_year.mean(), inplace=True)


# In[52]:


authors = {}
for i, line in enumerate(books.authors.unique()):
  temp = list(map(lambda x: x.strip(), line.split(',')))
  for author in temp:
    if not authors.get(author):
      authors[author] = [0] * len(books)
    authors[author][i] = 1


# In[53]:


pref = set()
for key in authors.keys():
  s = sum(authors[key])
  if s > 1:
    pref.add(key)
pref.add('Others')


# In[54]:


authors['Others'] = [0] * len(books)
d = authors.copy()
for author in d.keys():
  if author not in pref:
    for i in range(len(books)):
      if d[author][i]:
        authors['Others'] = 1
    del authors[author]


# In[55]:


authors = pd.DataFrame.from_dict(authors)
books.drop(labels=['authors'], axis=1, inplace=True)


# In[56]:


X = books.values
X = minmax_scale(X)
books = pd.DataFrame(X, columns=books.columns)
books = pd.concat((books, authors), axis=1)
book_features = sp.csr_matrix(books)


# In[57]:


tags = pd.read_csv('tags.csv')


# In[58]:


def delete_punctuation(text):
  exclude = string.punctuation + "«»\n-"
  table = str.maketrans(exclude, " " * len(exclude))
  return text.translate(table).lower()


# In[59]:


def format_str(text):
  words = [word for word in delete_punctuation(text).split()]
  return ' '.join(words)


# In[60]:


t = {}
for line in tags.values:
  if type(line[1]) == str:
    t[line[0]] = format_str(line[1])


# In[61]:


book_tags = pd.read_csv('book_tags.csv')


# In[62]:


g = {}
book_tags.head(50)
for i, line in enumerate(book_tags.values):
  if line[0] in g:
    if line[1] in t:
      if line[2] >= 500:
        g[line[0]] += ' ' + t[line[1]]
      else:
        del t[line[1]]
  else:
    if line[1] in t:
      if line[2] >= 500:
        g[line[0]] = t[line[1]]


# In[63]:


t = []
for i in gid:
  if i in g:
    t.append(g[i])
  else:
    t.append('')


# In[64]:


vectorizer = CountVectorizer()
vectorized_tags = vectorizer.fit_transform(t)
transformer = TfidfTransformer()
transformed_tags = transformer.fit_transform(vectorized_tags)


# In[65]:


book_features = sp.hstack((book_features, transformed_tags))


# In[67]:


model = LightFM(no_components=50, loss='warp', random_state=42)
model.fit(ratings, item_features=book_features, num_threads=cpu_count(), epochs=100)


# In[147]:


u = {}
for line in new_ratings.values:
  if u.get(line[0]):
    u[line[0]].add(line[1])
  else:
    u[line[0]] = set()
    u[line[0]].add(line[1])


# In[148]:


pred_u = []
pred_b = []
for key in u.keys():
  for i in range(10000):
    if i not in u[key]:
      pred_u.append(int(key))
      pred_b.append(i)


# In[149]:


preds = pd.DataFrame(data=zip(pred_u, pred_b), index=None, columns=['user_id', 'book_id'])
preds['score'] = model.predict(preds.user_id.values, preds.book_id.values, 
                      item_features=book_features)
preds.sort_values(by=['user_id', 'score'], ascending=[True, False], inplace=True)


# In[150]:


predictions = pd.DataFrame(columns=['user_id', 'book_id', 'score'])
for key in u.keys():
    predictions = pd.concat((predictions, preds.loc[preds['user_id'] == key].head(50)))


# In[151]:


predictions.drop(labels=['score'], axis=1, inplace=True)
predictions.to_csv('predictions.txt', index=False)


# In[ ]:




