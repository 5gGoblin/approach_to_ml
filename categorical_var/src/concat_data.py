import pandas as pd
from sklearn import preprocessing
import config

'''This concatenating will not work in a live setting.
You can either assume that your test data will have the same categories 
as training or you have to introduce a new 'unknown' category for 
unseen classes.'''

train = pd.read_csv(config.TRAINING_FILE)
test = pd.read_csv(config.TEST_FILE)

test['target'] = -1

#concat data
concat_data = pd.concat([train, test]).reset_index(drop = True)

features = [x for x in train.columns if x not in ['id', 'target']]

for feature in features:
    lbl_enc = preprocessing.LabelEncoder()

    temp_col = concat_data[feature].fillna("Unknown").astype(str).values
    concat_data[feature] = lbl_enc.fit_transform(temp_col)

#split data again
train = concat_data[concat_data.target != -1].reset_index(drop = True)
test = concat_data[concat_data.target == -1].reset_index(drop = True)

#print(train.ord_4.value_counts())

#set classes with less than 2000 members as rare
#This will allow for your model to work in a live setting
#it will set unseend classes as rare
train.loc[train["ord_4"].value_counts()[train["ord_4"]].values < 2000, "ord_4"] = "Rare"

print(train.ord_4.value_counts())
