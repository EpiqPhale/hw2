
import pandas as pd
import math

class Node:
    def __init__(self, threshold):
        self.threshold = threshold
        self.column = None
        self.label = None
        self.children = {}

    def makePrediction(self, data):
        if self.label is not None: #leaf node
            return self.label

        feature_value = data[self.column] #get the value from the data at the correct column
        if feature_value < self.threshold:
            return self.children['left'].makePrediction(data)
        else:
            return self.children['right'].makePrediction(data)


def entropy(set):
    #sum the third column and divid by the total number of rows to get the probability of "1"
    if len(set) == 0:
        return 0
    prob = sum(set.iloc[:, 2]) / len(set)
    if prob == 0 or prob == 1:  #prevent log(0) error
        return 0
    return -1 * prob * math.log2(prob) - (1 - prob) * math.log2(1 - prob)    #return the entropy

def getGainRatio(data, leftSplit, rightSplit, totalEntropy, weightedEntropy):
    # get info gain
    gain = totalEntropy - weightedEntropy

    # get split info
    leftProb = len(leftSplit) / len(data)
    rightProb = len(rightSplit) / len(data)

    if leftProb == 0:  # prevent log(0) error
        leftInfo = 0
    else:
        leftInfo = -1 * leftProb * math.log2(leftProb)

    if rightProb == 0:  # prevent log(0) error
        rightInfo = 0
    else:
        rightInfo = -1 * rightProb * math.log2(rightProb)

    splitInfo = leftInfo + rightInfo

    # get gain ratio
    if splitInfo == 0:  # prevent divide by 0 error
        return 0
    else:
        return gain / splitInfo

def findColumnSplit(data):
    #for each column
    #consider each value as a potential split
    #calculate the entropy for each split
    #calculate info gain for the split
    #select split with highest gain ratio
    #compare the candidate split from each column
    #return the column(feature) and the split value(threshold)
    bestGain = float('-inf')
    bestsplit = 0
    bestColumn = 0
    bestsplit = None

    #only 2 features, so only 2 columns
    for i in range(0, 2):
        data.sort_values(by=data.columns[i])

        #check every possible split (use every possible value for dimension criteria)
        for j in range(0, len(data)):
            leftSplit = data[data.iloc[:, i] < data.iloc[j, i]]
            rightSplit = data[data.iloc[:, i] >= data.iloc[j, i]] #x_j >= c criteria

            #get entropies
            leftEntropy = entropy(leftSplit)
            rightEntropy = entropy(rightSplit)
            totalEntropy = entropy(data)
            weightedEntropy = (len(leftSplit) / len(data)) * leftEntropy + (len(rightSplit) / len(data)) * rightEntropy

            #get gain ratio
            gainRatio = getGainRatio(data, leftSplit, rightSplit, totalEntropy, weightedEntropy)

            #finds the split with the highest gain ratio
            #implements arbitrary tie breaking by selecting the first split with the highest gain
            if gainRatio > bestGain:
                bestGain = gainRatio
                bestSplit = data.iloc[j, i]
                bestColumn = i

    return bestColumn, bestSplit, bestGain

def makeTree(data):
    #recursively build the tree

    #find the best split
    column, split, gain = findColumnSplit(data)
    #create a node
    node = Node(split)
    node.column = column
    #sort the data by the best column
    data.sort_values(by=data.columns[column])
    #if we meet the stopping criteria, return a leaf node

    #node is empty
    if len(data) == 0:
        node.label = 1
        return node
    #all splits have zero gain ratio
    if gain == 0:
        #return a leaf node with the most common label
        average = data.iloc[:, 2].mean()
        node.label = 1 if average >= 0.5 else 0 #>= because if no majority class, return 1
        return node
    #entropy of all candidate splits is zero
    if entropy(data) == 0:
        #return a leaf node with the most common label
        average = data.iloc[:, 2].mean()
        node.label = 1 if average >= 0.5 else 0
        return node

    #otherwise make left and right child sets
    leftSplit = data[data.iloc[:, column] < split]
    rightSplit = data[data.iloc[:, column] >= split]

    node.children['left'] = makeTree(leftSplit)
    node.children['right'] = makeTree(rightSplit)

    return node



# Main Program
#read in data
D1 = pd.read_csv('Homework 2 data/D1.txt', sep=' ')
D2 = pd.read_csv('Homework 2 data/D2.txt', sep=' ')
D3leaves = pd.read_csv('Homework 2 data/D3leaves.txt', sep=' ')
Dbig = pd.read_csv('Homework 2 data/Dbig.txt', sep=' ')
Druns = pd.read_csv('Homework 2 data/Druns.txt', sep=' ')

tree1 = makeTree(D1)
predictions = []

for index, row in D2.iterrows():
    prediction = tree1.makePrediction(row)
    predictions.append(prediction)

# Convert the list to a pandas Series if needed
predictions_series = pd.Series(predictions)

#print head
print(D1.head())