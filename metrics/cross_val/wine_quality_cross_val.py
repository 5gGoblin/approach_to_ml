import pandas as pd 
from sklearn import tree, metrics, model_selection
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rc('xtick', labelsize = 20)
matplotlib.rc('ytick', labelsize = 20)

data = pd.read_csv("../datasets/wineQualityReds.csv")

quality_mapping = {
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
}
data["quality"] = data.quality.astype(int).map(quality_mapping)

train_scores = [.5]
val_scores = [.5]
test_size = 1200
tree_depth = 20
num_of_samples = 10

cols = ['fixed.acidity', 'volatile.acidity', 'citric.acid',
        'residual.sugar', 'chlorides', 'free.sulfur.dioxide',
        'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol']

for depth in range(1, tree_depth + 1):
    train_sample_scores = [.5]
    val_sample_score = [.5]
    sample_num = 0
    while sample_num < num_of_samples:
        data = data.sample(frac = 1).reset_index(drop = True).copy()
        train = data.head(test_size)
        validation = data.tail(len(data)-test_size)

        clf = tree.DecisionTreeClassifier(max_depth = depth)

        clf.fit(train[cols], train.quality)
        train_predicitons = clf.predict(train[cols])
        val_predictions = clf.predict(validation[cols])

        train_acc = metrics.accuracy_score(
            train.quality, train_predicitons
        )
        val_acc = metrics.accuracy_score(
            validation.quality, val_predictions
        )

        train_sample_scores.append(train_acc)
        val_sample_score.append(val_acc)

        sample_num += 1
    train_scores.append(max(train_sample_scores))
    val_scores.append(max(val_sample_score))

plt.figure(figsize = (10, 5))
sns.set_style("whitegrid")
plt.plot(train_scores, label = "Train Accuracy")
plt.plot(val_scores, label = "Validation Accuracy")
plt.legend(loc = "upper left", prop = {'size' : 15})
plt.xticks(range(0, 26, 5))
plt.xlabel("Max Depth", size = 20)
plt.ylabel("Accuracy", size = 20)
plt.show()


print(f"Best Train Accuracy: {max(train_scores):.3f} \nBest Validation Accuracy: {max(val_scores):.3f}")
