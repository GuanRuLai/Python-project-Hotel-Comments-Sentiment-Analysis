from snownlp import SnowNLP
from snownlp import sentiment
from snownlp import seg
from lotecc import lote_chinese_conversion as lotecc
import pandas as pd

# change content's language into Traditional Chinese
converted = lotecc(conversion="s2twp",
                   input="C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\ChnSentiCorp_htl_all.csv",
                   output="C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\hotel_all.csv",
                   in_enc="utf-8",
                   out_enc="utf-8")

# check file
df = pd.read_csv("C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\hotel_all.csv")
# print(pd_all.head(10))

# check rows of the pos. and neg. datas
print("The numbers of positive comments: ", len(df[df["label"] == 1]), "rows")
print("The numbers of negative comments: ", len(df[df["label"] == 0]), "rows")

# gain positive dataset and split training and testing set
df_pos = df[df["label"] == 1]
df_pos = df_pos.sample(2444) # let the rows of pos and neg datasets be the same
df_pos_test = df_pos.iloc[:100]
# print(df_pos_test)
df_pos = df_pos.drop(columns="label") # keep "review" column
df_pos_train = df_pos.iloc[100:]
# print(df_pos_train)
df_pos_train.to_csv("C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\pos_train.csv",
                    header=False,
                    index=False)

# gain negative dataset and split training and testing set
df_neg = df[df["label"] == 0]
df_neg = df_neg.sample(frac=1.0) # let the datas be sorted randomly
df_neg_test = df_neg.iloc[:100]
# print(df_neg_test)
df_neg = df_neg.drop(columns="label") # keep "review" column
df_neg_train = df_neg.iloc[100:]
# print(df_neg_train)
df_neg_train.to_csv("C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\neg_train.csv",
                    header=False,
                    index=False)

# concatinate two testing sets
test_all = pd.concat([df_pos_test, df_neg_test], axis=0) # combind vertically
test_all = test_all.sample(frac=1.0) # let the datas be sorted randomly
test_all.to_csv("C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\test_all.csv",
                header=False,
                index=False)

# train self-customized model
sentiment.train("C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\neg_train.csv",
                "C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\pos_train.csv")
sentiment.save("C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\hotel_sentiment.marshal")

# test model
score = 0
with open("C:\\Users\\HP\\Desktop\\project\\NLP-Sentiment-Analysis\\test_all.csv", "r", encoding="utf-8") as f:
    datas = f.readlines()
    for data in datas:
        label = data.split(",")[0]
        text = data.split(",")[1]

        if SnowNLP(text).sentiments < 0.5:
            ss = 0
        else:
            ss = 1

        if int(label) == ss:
            score += 1

print("正確率: {}".format(score/len(datas)))