import pandas as pd
from sklearn import linear_model, metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:

    def evaluate_score(self, x, y):
        model = linear_model.LogisticRegression() #Select model that fits your needs
        model.fit(x, y)
        predictions = model.predict_proba(x)[:,1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def _feature_selection(self, x, y):
        good_features = []
        best_scores = []

        num_features = x.shape[1]

        while True:
            this_feature = None
            best_score = 0

            for feature in range(num_features):
                if feature in good_features:
                    continue
                selected_features = good_features + [feature]
                x_train = x[:,selected_features]
                score = self.evaluate_score(x_train, y)
                
                if score > best_score:
                    this_feature = feature
                    best_score = score

            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)

            #if we didnt improve scores, break while loop
            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break
                
        return best_scores[:-1], good_features[:-1]

    def __call__(self, x, y):
        scores, features = self._feature_selection(x, y)
        return x[:, features], scores

if __name__ == "__main__":
    x, y = make_classification(n_samples = 1000, n_features = 100)
    x_transformed, scores = GreedyFeatureSelection()(x,y)
    print(x_transformed)
    print(scores)

