import MeCab
import pandas as pd
from sklearn import model_selection, svm
from sklearn.feature_extraction.text import TfidfVectorizer


"""
Procedure
1. scrapingして文書データを抽出
2. 文書データをtfidfで特徴量に変換
3. LinearSVCでカテゴライズ
4. CrossValidationで検証
"""


def _prepare_data(df, categories):
    docs = df["Text"].tolist()
    types = df["Type"].tolist()
    labels = []
    for type in types:
        category_id = categories[type]
        labels.append(category_id)

    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(
        docs, labels, test_size=0.1, random_state=0
    )
    return train_data, test_data, train_labels, test_labels


def _doc2word_list(doc):
    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse("")
    node = tagger.parseToNode(doc)

    word_list = []
    while node:
        meta = node.feature.split(",")
        if meta[0] == "名詞":
            word_list.append(meta[6])
        node = node.next

    return word_list


def _get_vectorized_data(train_data, test_data):
    vectorizer = TfidfVectorizer(tokenizer=_doc2word_list, min_df=0.03, max_df=0.5)
    train_matrix = vectorizer.fit_transform(train_data)
    test_matrix = vectorizer.transform(test_data)
    return train_matrix, test_matrix


def _prepare_clfmodel(train_matrix, train_labels):
    clf = svm.LinearSVC(loss="hinge", C=1.0, class_weight="balanced", random_state=0)
    clf.fit(train_matrix, train_labels)
    return clf


def _get_category_name(categories, category_id):
    return [k for k, v in categories.items() if v == category_id]


def predict(input_filepath: str) -> None:
    # Text,Typeの2カラムで構成されたcsvを読み込む
    df = pd.read_csv(input_filepath)
    # categoryはハードコーディング
    categories = {"Howto": 1, "Interview": 2, "Growth": 3, "Tech": 4, "NUR": 5}

    train_data, test_data, train_labels, test_labels = _prepare_data(df, categories)

    """
    test_dataを手で入れる場合
    test_data = [
        "この記事の目次1 LINE連携の基本1.1 LINEが提供しているもの1.2 Messagin...",
        "この記事の目次1 初心者向け1.1 UnityではじめるC# 基礎編1.2 ゲームプロ...
    ]
    """

    train_matrix, test_matrix = _get_vectorized_data(train_data, test_data)
    clf = _prepare_clfmodel(train_matrix, train_labels)

    print("Train accuracy = %.3f" % clf.score(train_matrix, train_labels))
    print("Test accuracy = %.3f" % clf.score(test_matrix, test_labels))

    preds = clf.predict(test_matrix)
    for i, pred in enumerate(preds):
        print(_get_category_name(categories, pred), test_data[i])


if __name__ == "__main__":
    predict()
