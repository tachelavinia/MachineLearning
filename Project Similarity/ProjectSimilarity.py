from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
import scipy as sp
import scipy.linalg as sp_linalg

vectorizer = CountVectorizer(min_df= 1, stop_words='english')

# print(vectorizer)
content = ["How to format my hard disk","Hard disk format problems"]
X = vectorizer.fit_transform(content)
vectorizer.get_feature_names()

DIR = "./data/toy"
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]
# print (posts)

X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

def dist_raw(v1, v2):
    delta = v1-v2
    return sp_linalg.norm(delta.toarray())

def dist_norm(v1, v2):
    v1_normalized = v1/sp_linalg.norm(v1.toarray())
    v2_normalized = v2/sp_linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp_linalg.norm(delta.toarray())

best_doc = None
best_dist = np.inf
best_i = None
for i in range(0, num_samples):
    post = posts[i]
    if post==new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec, new_post_vec)
    print("=== Post %i with dist=%.2f: %s"%(i, d, post))
    if d<best_dist:
        best_dist = d
        best_i = i
print("Best post is %i with dist=%.2f"%(best_i, best_dist))

