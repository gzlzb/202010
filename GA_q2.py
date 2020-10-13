import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from deap import creator, base, tools, algorithms
#from scoop import futures
from sklearn import linear_model
import statsmodels.formula.api as smf
import random
import numpy
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.decomposition import PCA


# Read in data from CSV
# Data set from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
Correct = pd.read_csv('Correct_15.csv', sep=',')
Predict = pd.read_csv('Predict_15.csv', sep=',')
X_train = Correct.drop(['LogBIO'], axis=1)
X_test = Predict.drop(['LogBIO'], axis=1)
data = pd.concat([X_train,X_test])
sample_size = Correct.shape[0]

pca = PCA(sample_size)  #选取n个主成分
pca.fit(data)
#降低维度
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train['LogBIO'] = Correct['LogBIO']
X_test['LogBIO'] = Predict['LogBIO']
Correct, Predict = X_train, X_test



# Encode the classification labels to numbers
# Get classes and one hot encoded feature vectors
#le = LabelEncoder()
#le.fit(Correct['LogBIO'])
#y = Correct.LogBIO
#y_train = le.transform(Correct['LogBIO'])
y_train = Correct.LogBIO
X_train = Correct.drop(['LogBIO'], axis=1)
y_test = Predict.LogBIO
X_test = Correct.drop(['LogBIO'], axis=1)

# Form training, test, and validation sets
#X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split(allFeatures, allClasses, test_size=0.20, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20, random_state=42)

def q_squared(y_true, X):
    #n = len(X)
    #y = data.LogBIO
    #X = data.drop([response], axis=1)
    lr = LinearRegression()
    import warnings
    warnings.filterwarnings("ignore")
    #lr = SVR()
    y_pred = cross_val_predict(lr, X, y_true, cv = 4)
    """lr = linear_model.LinearRegression(fit_intercept=False)
    predicted0 = cross_val_predict(lr, X, y, cv = n)"""
    #cv = pd.DataFrame({'y':y, 'y1':predicted})
    """cv = pd.DataFrame({'y':y, 'y1':predicted, 'y0':predicted0})
    cv = pd.DataFrame({'y':y, 'y0':predicted0})"""
    #r2 = smf.ols('y~y1', cv).fit().rsquared_adj
    """r02 = smf.ols('y~y0',cv).fit().rsquared_adj"""
                
    """import math
    rm2 = r2 * math.sqrt(abs(1-abs(r2-r02)))"""
    #r, p = stats.pearsonr(y, y_pred)
    #SSRes = numpy.sum((y_true-y_pred)**2)
    #SSRes = SSRes/len(y_true)
    #SStot = numpy.sum((y_true-numpy.mean(y_true))**2)
    #SStot = SStot/len(y_true)
    #import scipy.stats as stats
    r, p = stats.pearsonr(y_true, y_pred)
    return r * r

def q2ven(cor):

    clf = LinearRegression()
    #clf = SVR(gamma='scale')
    
    val = pd.DataFrame(columns=['ture', 'pred'])
    import warnings
    warnings.filterwarnings("ignore")

    fold = 10
    for group in range(fold):
        va = cor[cor.index%fold==group]
        tr = cor[cor.index%fold!=group]
        y_true = tr.LogBIO
        X = tr.drop(["LogBIO"], axis=1)
        clf.fit(X, y_true)
        y_true = va.LogBIO
        X = va.drop(["LogBIO"], axis=1)
        y_pred = clf.predict(X)
        cv = pd.DataFrame({'true':y_true, 'pred':y_pred})
        val = val.append(cv)

    y_true = val.true
    y_pred = val.pred
    
    import scipy.stats as stats
    corr, p = stats.pearsonr(y_true, y_pred)
    #PRESS = numpy.sum(y_true-y_pred)**2
    #TSS = numpy.sum((y_true-numpy.mean(y_true))**2)

    return corr


# Feature subset fitness function
def getFitness(individual, X_train, y_train):

	# Parse our feature columns that we don't use
	# Apply one hot encoding to the features
	cols = [index for index in range(len(individual)) if individual[index] == 0]
	X_trainParsed = X_train.drop(X_train.columns[cols], axis=1)
	X_trainOhFeatures = pd.get_dummies(X_trainParsed)
	X_trainOhFeatures['LogBIO'] = y_train
	#X_testParsed = X_test.drop(X_test.columns[cols], axis=1)
	#X_testOhFeatures = pd.get_dummies(X_testParsed)

	# Remove any columns that aren't in both the training and test sets
	'''sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
	removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
	removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
	X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
	X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)'''

	# Apply logistic regression on the data, and calculate accuracy
	#clf = LogisticRegression()
	#clf.fit(X_trainOhFeatures, y_train)
	#predictions = clf.predict(X_testOhFeatures)
	accuracy = q2ven(X_trainOhFeatures)

	# Return calculated accuracy as fitness
	return (accuracy,)

#========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(Correct.columns) - 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness, X_train=X_train, y_train=y_train)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#========

def getHof():

	# Initialize variables to use eaSimple
	numPop = sample_size
	numGen = 30
	pop = toolbox.population(n=numPop)
	hof = tools.HallOfFame(numPop * numGen)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	# Launch genetic algorithm
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

	# Return the hall of fame
	return hof

def getMetrics(hof):

	# Get list of percentiles in the hall of fame
	percentileList = [i / (len(hof) - 1) for i in range(len(hof))]
	
	# Gather fitness data from each percentile
	testAccuracyList = []
	validationAccuracyList = []
	individualList = []
	for individual in hof:
		testAccuracy = individual.fitness.values
		validationAccuracy = getFitness(individual, X_train,y_train)
		testAccuracyList.append(testAccuracy[0])
		validationAccuracyList.append(validationAccuracy[0])
		individualList.append(individual)
	testAccuracyList.reverse()
	validationAccuracyList.reverse()
	return testAccuracyList, validationAccuracyList, individualList, percentileList


if __name__ == '__main__':

	'''
	First, we will apply logistic regression using all the features to acquire a baseline accuracy.
	'''
	individual = [1 for i in range(len(X_train.columns))]
	testAccuracy = getFitness(individual, X_train, y_train)
	validationAccuracy = getFitness(individual, X_train, y_train)
	print('\nTest accuracy with all features: \t' + str(testAccuracy[0]))
	print('Validation accuracy with all features: \t' + str(validationAccuracy[0]) + '\n')

	'''
	Now, we will apply a genetic algorithm to choose a subset of features that gives a better accuracy than the baseline.
	'''
	hof = getHof()
	testAccuracyList, validationAccuracyList, individualList, percentileList = getMetrics(hof)

	# Get a list of subsets that performed best on validation data
	maxValAccSubsetIndicies = [index for index in range(len(validationAccuracyList)) if validationAccuracyList[index] == max(validationAccuracyList)]
	maxValIndividuals = [individualList[index] for index in maxValAccSubsetIndicies]
	maxValSubsets = [[list(X_train)[index] for index in range(len(individual)) if individual[index] == 1] for individual in maxValIndividuals]

	print('\n---Optimal Feature Subset(s)---\n')
	for index in range(len(maxValAccSubsetIndicies)):
		print('Percentile: \t\t\t' + str(percentileList[maxValAccSubsetIndicies[index]]))
		print('Validation Accuracy: \t\t' + str(validationAccuracyList[maxValAccSubsetIndicies[index]]))
		print('Individual: \t' + str(maxValIndividuals[index]))
		print('Number Features In Subset: \t' + str(len(maxValSubsets[index])))
		print('Feature Subset: ' + str(maxValSubsets[index]))

	Cor = Correct[maxValSubsets[index]]
	Cor['LogBIO'] = y_train
	Cor.to_csv("Correct_2.csv", index=False)
	Pre = Predict[maxValSubsets[index]]
	Pre['LogBIO'] = y_test
	Pre.to_csv("Predict_2.csv", index=False)

	'''
	Now, we plot the test and validation classification accuracy to see how these numbers change as we move from our worst feature subsets to the 
	best feature subsets found by the genetic algorithm.
	'''
	'''# Calculate best fit line for validation classification accuracy (non-linear)
	tck = interpolate.splrep(percentileList, validationAccuracyList, s=5.0)
	ynew = interpolate.splev(percentileList, tck)

	e = plt.figure(1)
	plt.plot(percentileList, validationAccuracyList, marker='o', color='r')
	plt.plot(percentileList, ynew, color='b')
	plt.title('Validation Set Classification Accuracy vs. \n Continuum with Cubic-Spline Interpolation')
	plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
	plt.ylabel('Validation Set Accuracy')
	e.show()

	f = plt.figure(2)
	plt.scatter(percentileList, validationAccuracyList)
	plt.title('Validation Set Classification Accuracy vs. Continuum')
	plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
	plt.ylabel('Validation Set Accuracy')
	f.show()

	g = plt.figure(3)
	plt.scatter(percentileList, testAccuracyList)
	plt.title('Test Set Classification Accuracy vs. Continuum')
	plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
	plt.ylabel('Test Set Accuracy')
	g.show()

	input()'''