import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the player stats data from a CSV file
data = pd.read_csv('per_game_stats.csv')

# Separate features and target variable
X = data.drop(['Player','Rk', 'Pos', 'Tm', 'AllStar'], axis=1)
y = data['AllStar']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a random forest classifier
rf = RandomForestClassifier()
rf.fit(X, y)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Perform k-fold cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)  # Set the number of folds (cv) to 5, adjust as needed

# Print the cross-validation scores
print("Cross-validation scores:")
print(cv_scores)
print("Average accuracy: {:.2f}".format(cv_scores.mean()))

# Train the random forest classifier on the entire dataset
#rf.fit(X, y)

# Predict All-Star players for a new season
new_season_data = pd.read_csv('2023-predict.csv')
new_season_players = new_season_data['Player']
new_season_X = new_season_data.drop(['Player','Rk', 'Pos', 'Tm'], axis=1)

all_star_predictions = rf.predict(new_season_X)
all_star_players = new_season_players[all_star_predictions == 1]

print("Predicted All-Star Players:")
for player in all_star_players:
    print(player)

#Plotting decision tree (out of scope quickly)
fig = plt.figure(figsize=(15, 10))
plot_tree(rf.estimators_[0], max_depth = 3,
          filled=True, impurity=True, feature_names = X.columns.values.tolist(),
          rounded=True)
plt.show()
fig.savefig('figure_name.png')

# look at the feature importances
dfFeatures = pd.DataFrame({'Features':X.columns.values.tolist(),'Importances':rf.feature_importances_})
print(dfFeatures.sort_values(by='Importances',ascending=False))