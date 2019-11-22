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


def prepare_data(df, categories):
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


def doc2word_list(doc):
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


def get_vectorized_data(train_data, test_data):
    vectorizer = TfidfVectorizer(tokenizer=doc2word_list, min_df=0.03, max_df=0.5)
    train_matrix = vectorizer.fit_transform(train_data)
    test_matrix = vectorizer.transform(test_data)
    return train_matrix, test_matrix


def prepare_clfmodel(train_matrix, train_labels):
    clf = svm.LinearSVC(loss="hinge", C=1.0, class_weight="balanced", random_state=0)
    clf.fit(train_matrix, train_labels)
    return clf


def get_category_name(categories, category_id):
    return [k for k, v in categories.items() if v == category_id]


def main():
    # Text,Typeの2カラムで構成されたcsvを読み込む
    df = pd.read_csv(
        "/Users/takuma.nishi/projects/ml_tools/classification/text_type.csv"
    )
    # categoryはハードコーディング
    categories = {"Howto": 1, "Interview": 2, "Growth": 3, "Tech": 4, "NUR": 5}

    train_data, test_data, train_labels, test_labels = prepare_data(df, categories)

    """     
    test_dataを手で入れる場合
    test_data = [
        "この記事の目次1 LINE連携の基本1.1 LINEが提供しているもの1.2 MessagingAPIについて2 MessagingAPIの基礎知識2.1 基本メッセージ機能2.2 マルチメディア機能2.3 その他、便利機能3 MessagingAPIの使い方3.1 コンソールの利用方法4 まとめ",
        "この記事の目次1 初心者向け1.1 UnityではじめるC# 基礎編1.2 ゲームプログラマになる前に覚えておきたい技術2 中級者向け2.1 Unityゲーム開発 オンライン3Dアクションゲームの作り方2.2 Android UI Cookbook for 4.0　ICS（Ice Cream Sandwich）アプリ開発術3 上級者向け3.1 モバイルアプリ開発エキスパート養成読本 (Software Design plus)3.2 Unity ゲームエフェクト入門 Shurikenで作る! ユーザーを引き込む演出手法4 まとめ",
    ] 
    """

    train_matrix, test_matrix = get_vectorized_data(train_data, test_data)
    clf = prepare_clfmodel(train_matrix, train_labels)

    print("Train accuracy = %.3f" % clf.score(train_matrix, train_labels))
    print("Test accuracy = %.3f" % clf.score(test_matrix, test_labels))

    preds = clf.predict(test_matrix)
    for i, pred in enumerate(preds):
        print(get_category_name(categories, pred), test_data[i])


if __name__ == "__main__":
    main()
