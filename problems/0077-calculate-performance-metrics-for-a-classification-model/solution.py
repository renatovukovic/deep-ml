
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here

	#get true positives, negatives and false positives and negatives
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for label, pred in zip(actual, predicted):
		if label:
			if pred:
				tp += 1
			else:
				fn += 1
		else:
			if pred: 
				fp += 1
			else:
				tn += 1
	
	confusion_matrix = [[tp, fn], [fp, tn]]

	accuracy = (tp + tn) / (tp + fp + tn + fn)

	f1 = 2 * tp / (2 * tp + fp + fn)

	#specificity can be seen as negative recall
	specificity = tn / (tn + fp)
	#negative predictive value is negative precision
	negativePredictive = tn / (tn + fn)


	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
