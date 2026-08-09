from sklearn.feature_selection import \
    chi2,f_classif, f_regression, mutual_info_classif, mutual_info_regression,SelectKBest,SelectPercentile

class UnivariateFeatureSelection:
    def _init_(self, n_features, problem_type, scoring):

        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info-regression": mutual_info_classif
            }

        #raise exception if no valid scoring method
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")

        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k = n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile = int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")
    
    def fit(self, x, y):
        return self.selection.fit(x, y)

    def transform(self, x):
        return self.selection.transform(x)

    def fit_transform(self, x, y):
        return self.selection.fit_transform(x, y)

