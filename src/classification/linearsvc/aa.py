from sklearn.datasets import fetch_20newsgroups

categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
twenty_train = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)

twenty_train.target_names
len(twenty_train.data)
len(twenty_train.filenames)

print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])

print(twenty_train.target[:10])

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

count_vect.vocabulary_.get(u"algorithm")

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
