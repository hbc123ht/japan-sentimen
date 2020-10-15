import os 
import json 
import pandas as pd 
from sklearn.utils import shuffle

data = {
    "text": [],
    "label": []
}

all_positive = open("data/data1/10000positive.txt", encoding="utf8").readlines()
for x in all_positive: 
    data['text'].append(x)
    data['label'].append("__label__positive")

all_negative = open("data/data1/10000negative.txt", encoding="utf8").readlines()
for x in all_negative: 
    data['text'].append(x)
    data['label'].append("__label__negative")

df = pd.DataFrame(data=data)
df = shuffle(df)

# df.iloc[0:int(len(data)*0.8)].to_csv('data/train/train.csv', sep='\t', index = False, header = False)
# df.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('data/train/test.csv', sep='\t', index = False, header = False)
# df.iloc[int(len(data)*0.9):].to_csv('data/train/dev.csv', sep='\t', index = False, header = False)

data_train = df.iloc[0:int(len(df)*0.8)]
data_test = df.iloc[int(len(df)*0.8):int(len(df)*0.9)]
data_dev = df.iloc[int(len(df)*0.9):]

def write_df_to_txt(path, the_df): 
    file = open(path, "w", encoding="utf8")

    samples =  []
    for x, y in zip(the_df['label'], the_df['text']): 
        samples.append("{0}\t{1}".format(x, y))

    file.writelines(samples)

write_df_to_txt('data/train/train.txt', data_train)
write_df_to_txt('data/train/text.txt', data_test)
write_df_to_txt('data/train/dev.txt', data_dev)
