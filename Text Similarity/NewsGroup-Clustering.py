import sklearn.datasets
import nltk.stem
import scipy.linalg as sp_linalg
from sklearn.feature_extraction.text import TfidfVectorizer

MLCOMP_DIR = r"./data"
data = sklearn.datasets.load_mlcomp("20news-18828", mlcomp_root=MLCOMP_DIR)

print(data.filenames)
print(len(data.filenames))
print(data.target_names)

#train_data = sklearn.datasets.load_mlcomp("20news-18828", "train",mlcomp_root=MLCOMP_DIR)
#print(len(train_data.filenames))
#
#test_data = sklearn.datasets.load_mlcomp("20news-18828","test", mlcomp_root=MLCOMP_DIR)
#print(len(test_data.filenames))

groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.ma c.hardware', 'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.load_mlcomp("20news-18828", "train",mlcomp_root=MLCOMP_DIR, categories=groups)
print(len(train_data.filenames))

english_stemmer = nltk.stem.SnowballStemmer('english')
#TF-IDF
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,stop_words='english',decode_error='ignore')
vectorized = vectorizer.fit_transform(train_data.data)


num_samples, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

num_clusters = 50
from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters, init='random', n_init=1,verbose=1)
km.fit(vectorized)

new_post = "Disk drive problems. Hi, I have a problem with my hard disk.After 1 year it is working only sporadically now.I tried to format it, but now it doesn't boot any more.Any ideas? Thanks."
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
similar_indices = (km.labels_==new_post_label).nonzero()[0]

similar = []
for i in similar_indices:
    dist = sp_linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, data.data[i]))
similar = sorted(similar)
print(len(similar))

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar)/2)]
show_at_3 = similar[-1]
print(show_at_1)
print(show_at_2)
print(show_at_3)