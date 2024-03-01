import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree, export_graphviz
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore" , category=DeprecationWarning)  #

#load the datasets
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')

cols = training.columns
cols = cols[:-1]

x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

#encode strings to numbers using LabelEncoder (0 and 1)
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

#Split your data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

#model
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
#print (clf1. score(x_train, y_train))
#print ("cross result========")
#scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=3)
#print (scores)
#print (scores.mean())
#print(clf.score(testx,testy))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

#feature_importances
#for f in range(10):
#    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))

print("Please reply Yes or No for the following symptoms")
def print_disease(node):
    #print "Symptom: ", node
    node = node[0]
    #print (len(node))
    val = node.nonzero()
    #print (val)
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree, feature_names):
    tree = tree.tree_
    #print(tree)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]
    #print ("def tree({}):".format(", ".join(feature_names)))
    symptoms_present = []
    def recurse(node, depth):
        indent = " " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_. threshold[node]
            print(name + " ?")
            ans = input()
            ans = ans.lower()
            if ans == 'yes':
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            print ("You may have "+ present_disease)
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("symptoms present " + str(list(symptoms_present)))
            print("symptoms given " + str(list(symptoms_given)))
            confidence_level = (1.0*len(symptoms_present))/ len(symptoms_given)
            print("Confidence level is " + str(confidence_level))
            
    recurse(0,1)
    
tree_to_code(clf,cols)
            