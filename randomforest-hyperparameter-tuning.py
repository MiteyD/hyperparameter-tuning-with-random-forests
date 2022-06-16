# This code is exactly the same as what we have done in the previous exercises. You do not need to read it again.

import pandas
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/san_fran_crime.csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import graphing # custom graphing code. See our GitHub repo for details

#Import the data from the .csv file
dataset = pandas.read_csv('san_fran_crime.csv', delimiter="\t")

# One-hot encode features
dataset = pandas.get_dummies(dataset, columns=["Category", "PdDistrict"], drop_first=False)

features = [c for c in dataset.columns if c != "Resolution"]

# Make a utility method that we can re-use throughout this exercise
# To easily fit and test out model
def fit_and_test_model(model):
    '''
    Trains a model and tests it against both train and test sets
    '''  
    global features

    # Train the model
    model.fit(train[features], train.Resolution)

    # Assess its performance
    # -- Train
    predictions = model.predict(train[features])
    train_accuracy = balanced_accuracy_score(train.Resolution, predictions)

    # -- Test
    predictions = model.predict(test[features])
    test_accuracy = balanced_accuracy_score(test.Resolution, predictions)

    return train_accuracy, test_accuracy


print("Ready!")
dataset.head()


# Split the dataset in an 90/10 train/test ratio. 
train, test = train_test_split(dataset, test_size=0.1, random_state=2, shuffle=True)


"""
Criteria to split on
The first hyperparameter we will work with is the criterion. This is essentially a kind of cost function that is used to determine whether a node should be split or not. We have two options available in the package that we are using: gini and entropy. Let's try them both:
"""



from sklearn.ensemble import RandomForestClassifier

# Shrink the training set temporarily to explore this
# setting with a more normal sample size
sample_size = 1000
full_trainset = train
train = full_trainset[:sample_size]


# Prepare the model 
rf = RandomForestClassifier(n_estimators=10,
                            # max_depth=12,
                            # max_features=cur_max_features,
                            random_state=2,
                            criterion="gini", 
                            verbose=False)
# Train and test the result
train_accuracy, test_accuracy = fit_and_test_model(rf)
# Train and test the result
print(train_accuracy, test_accuracy)

# Prepare the model 
rf = RandomForestClassifier(n_estimators=10,
                            random_state=2,
                            criterion="entropy", 
                            verbose=False)
# Train and test the result
train_accuracy, test_accuracy = fit_and_test_model(rf)
# Train and test the result
print(train_accuracy, test_accuracy)

# Roll back the train dataset to the full train set
train = full_trainset



"""
Minimum impurity decrease
The minimum impurity decrease is another criterion that is used to assess whether a node should be split. It is used by the gini or entropy algorithms we used, above. If minimu impurity decrease is high, then splitting a node must result in substantial performance improvement. If it is very low, then nodes can be split even if they offer very little to no performance improvements on the training dataset.
"""


import numpy as np

# Shrink the training set temporarily to explore this
# setting with a more normal sample size
full_trainset = train
train = full_trainset[:1000] # limit to 1000 samples

min_impurity_decreases = np.linspace(0, 0.0005, num=100)

# Train our models and report their performance
train_accuracies = []
test_accuracies = []

print("Working...")
for min_impurity_decrease in min_impurity_decreases:

    # Prepare the model 
    rf = RandomForestClassifier(n_estimators=10,
                                min_impurity_decrease=min_impurity_decrease,
                                random_state=2, 
                                verbose=False)
    
    # Train and test the result
    train_accuracy, test_accuracy = fit_and_test_model(rf)

    # Save the results
    test_accuracies.append(test_accuracy)
    train_accuracies.append(train_accuracy)


# Plot results
graphing.line_2D(dict(Train=train_accuracies, Test=test_accuracies), 
                    min_impurity_decreases,
                    label_x="Minimum impurity decreases (min_impurity_decrease)",
                    label_y="Accuracy",
                    title="Performance", show=True)

# Roll back the train dataset to the full train set
train = full_trainset


"""
Maximum number of features
When trees are created, they are provided with a subset of the data. This not only means they see a certain collection of rows (samples), but also a certain collection of columns (features). The more features are provided, the more likely a given tree is going to overfit. Let's see what happens when we restrict the maximum number of features that can be provided to each tree in the forest:
"""


# Shrink the training set temporarily to explore this
# setting with a more normal sample size
full_trainset = train
train = full_trainset[:1000] # limit to 1000 samples

max_features = range(10, len(features) +1)

# Train our models and report their performance
train_accuracies = []
test_accuracies = []

print("Working...")
for cur_max_features in max_features:
    # Prepare the model 
    rf = RandomForestClassifier(n_estimators=50,
                                max_depth=12,
                                max_features=cur_max_features,
                                random_state=2, 
                                verbose=False)
    
    # Train and test the result
    train_accuracy, test_accuracy = fit_and_test_model(rf)

    # Save the results
    test_accuracies.append(test_accuracy)
    train_accuracies.append(train_accuracy)


# Plot results
graphing.line_2D(dict(Train=train_accuracies, Test=test_accuracies), 
                    max_features,
                    label_x="Maximum number of features (max_features)",
                    label_y="Accuracy",
                    title="Performance", show=True)

# Roll back the trainset to the full set
train = full_trainset



"""
Seeding
Finally, we come to seeding. When trees are initially made, there is a degree of randomness that is used to decide which features and samples will be provided to which trees. Changing the random state (seed) value changes this initial state.

The random seed is not a parameter to be tuned, but its effects on our models shouldn't be forgotten, particularly when there isn't much data to work with. Let's see how our model behaves with different random states.
"""



# Shrink the training set temporarily to explore this
# setting with a more normal sample size
sample_size = 1000
full_trainset = train
train = full_trainset[:sample_size] 


seeds = range(0,101)

# Train our models and report their performance
train_accuracies = []
test_accuracies = []

for seed in seeds:
    # Prepare the model 
    rf = RandomForestClassifier(n_estimators=10,
                                random_state=seed, 
                                verbose=False)
    
    # Train and test the result
    train_accuracy, test_accuracy = fit_and_test_model(rf)

    # Save the results
    test_accuracies.append(test_accuracy)
    train_accuracies.append(train_accuracy)


# Plot results
graphing.line_2D(dict(Train=train_accuracies, Test=test_accuracies), 
                    seeds,
                    label_x="Seed value (random_state)",
                    label_y="Accuracy",
                    title="Performance", show=True)

# Roll back the trainset to the full set
train = full_trainset
