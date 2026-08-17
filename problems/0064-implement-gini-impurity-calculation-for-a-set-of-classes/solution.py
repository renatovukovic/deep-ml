
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	

	#get the number of occurences for each class
	occurences_dict = {}
	for entry in y:
		if entry not in occurences_dict:
			occurences_dict[entry] = 1
		else:
			occurences_dict[entry] += 1

	#get ratio of occurences squared for each class and sum
	class_sum = sum([(occurences_dict[entry] / len(y))**2 for entry in occurences_dict])

	gini = 1 - class_sum

	return round(gini, 3)