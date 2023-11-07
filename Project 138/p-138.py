import pandas as pd

df1 = pd.read_csv("shared_articles.csv")

df2 = pd.read_csv("users_interactions.csv")


print(df1.head(5))

print(df2.head(5))


# ---------------------------------- project 139 ---------------------------------------------------

print(df1[['title', 'eventType']].head(10))

df1 = df1[df1['eventType'] == 'CONTENT SHARED']

print(df1.head())

def find_total_events(df1_row):
  total_likes = df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "LIKE")].shape[0]
  total_views = df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "VIEW")].shape[0]
  total_bookmarks = df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "BOOKMARK")].shape[0]
  total_follows = df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "FOLLOW")].shape[0]
  total_comments = df2[(df2["contentId"] == df1_row["contentId"]) & (df2["eventType"] == "COMMENT CREATED")].shape[0]
  
  return total_likes + total_views + total_bookmarks + total_follows + total_comments

df1["total_events"] = df1.apply(find_total_events, axis=1)

df1 = df1.sort_values(['total_events'], ascending=[False])

print(df1[['title', 'eventType']].head(10))

# ---------------------------------- project 140 ---------------------------------------------------

def convert_lower_case(x):
  if isinstance(x , str):
    return x.lower()
  
  else:
    return ''

df1["title"] = df1["title"].apply(convert_lower_case)

df1.head()

# Import the countvectorizer class to create vectors.

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df1['title'])


# create the cosine similarity classifier.

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# reset the index.
df1 = df1.reset_index()
indices = pd.Series(df1.index, index=df1['contentId'])


def get_recommendations(contentId, cosine_sim):
    idx = indices[contentId]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df1['contentId'].iloc[movie_indices]