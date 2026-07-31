import pandas as pd
import config

data = pd.read_csv(config.TRAINING_FILE)


print(data.columns)
print(set(data.ord_2.values))


mapping = {
    "Freezing": 0,
    "Warm": 1,
    "Cold": 2,
    "Boiling Hot": 3,
    "Hot": 4,
    "Lava Hot": 5,
    "NONE": 6
}

data["ord_2"] = data.ord_2.fillna("NONE").map(mapping)

print(data["ord_2"].value_counts())
print(data.shape)

