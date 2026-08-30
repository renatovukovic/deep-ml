def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	#get the true positives, false positves, and false negatives
	tp = 0
	fp = 0
	fn = 0
	for true, pred in zip(y_true, y_pred):
		if true and pred:
			tp += 1
		elif true and not pred:
			fn += 1
		elif not true and pred:
			fp += 1

	f1 = 2*tp / (2*tp + fp + fn)
	
	return round(f1,3)