import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

X_train = np.loadtxt("../data/scaled_data/X_train.txt")
y_train = np.loadtxt("../data/scaled_data/y_train.txt")
X_test = np.loadtxt("../data/scaled_data/X_test.txt")

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'max_depth': [10, 20, None],
    'max_features': ['auto','log2', 'sqrt'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

with open('best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

with open('best_score.txt', 'w') as f:
    f.write(f"Best Score: {best_score:.4f}\n")

print("Tuning complete. Results saved:")
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score:.4f}")
