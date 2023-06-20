import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Load the player stats data from a CSV file
data = pd.read_csv('player_stats.csv')

# Separate features and target variable
X = data.drop(['Player', 'Pos', 'Tm', 'AllStar'], axis=1)
y = data['AllStar']

# Create a random forest classifier
clf = RandomForestClassifier()

# Perform k-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)  # Set the number of folds (cv) to 5, adjust as needed

# Print the cross-validation scores
print("Cross-validation scores:")
print(cv_scores)
print("Average accuracy: {:.2f}".format(cv_scores.mean()))

# Train the random forest classifier on the entire dataset
clf.fit(X, y)

# Predict All-Star players for a new season
new_season_data = pd.read_csv('new_season_player_stats.csv')
new_season_players = new_season_data['Player']
new_season_X = new_season_data.drop(['Player', 'Player-additional', 'Pos', 'Tm'], axis=1)

all_star_predictions = clf.predict(new_season_X)
all_star_players = new_season_players[all_star_predictions == 1]

print("Predicted All-Star Players:")
for player in all_star_players:
    print(player)