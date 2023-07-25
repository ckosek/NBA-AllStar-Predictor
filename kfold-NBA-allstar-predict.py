import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import classification_report,accuracy_score

# Load the player stats data from a CSV file
data = pd.read_csv('per_game_stats.csv')

# Separate features and target variable
X = data.drop(['Player', 'Pos', 'Tm', 'AllStar'], axis=1)
y = data['AllStar']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a random forest classifier
clf = RandomForestClassifier()
clf.fit(X, y)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Perform k-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)  # Set the number of folds (cv) to 5, adjust as needed

# Print the cross-validation scores
print("Cross-validation scores:")
print(cv_scores)
print("Average accuracy: {:.2f}".format(cv_scores.mean()))

# Train the random forest classifier on the entire dataset
#clf.fit(X, y)

# Predict All-Star players for a new season
new_season_data = pd.read_csv('2023-predict.csv')
new_season_players = new_season_data['Player']
new_season_X = new_season_data.drop(['Player', 'Pos', 'Tm'], axis=1)

all_star_predictions = clf.predict(new_season_X)
all_star_players = new_season_players[all_star_predictions == 1]

print("Predicted All-Star Players:")
for player in all_star_players:
    print(player)