from k_means import kMeans
from decision_tree import dTree
from svm import support_vector_machines
from naiveBayes import nBayes

def print_menu():
	print(' MENU')
	print(' 1: Income-Age grouping using K-Means Clustering')
	print(' 2: Spam Filter using Naive Bayes')
	print(' 3: Decision making for candidate via Resume using Decision Tree')
	print(' 4: Predicting the group of people based on previous income-age class using Support Vector Machines')
	print(' 5: Exit')

loop=True

while loop:
	print_menu()

	try:
		choice = int(input(' Enter your choice [1-5]: '))
	except Exception:
		print(' Enter valid choice!')
		choice = 6

	if choice == 1:
		print('\n K-Means Clustering technique is used to group unrealated data.'
				'\n For example : Unsupervised learning model (i.e. no additional information provided to work on) like where rich people lives')
		input(' Press enter to proceed to plot')
		kMeans()

	elif choice == 2:
		print('\n Naive Bayes is used for classifying data.'
				'\n For example : Supervised learning model (i.e. making action according to previous deduction) like drug-test results')
		input(' Press enter to proceed to result')
		nBayes()

	elif choice == 3:
		print('\n Decision Tree is used for resolving to get conclusion from past decisions.'
				'\n For example : Supervised learning model (i.e. making action according to previous deduction) like Should i play outside ?')
		input(' Press enter to proceed to get decision tree')
		dTree()

	elif choice == 4:
		print('\n Support Vector Machine is based on classifying higher dimensional data.'
				'\n For example : Supervised learning model (i.e. making action according to previous deduction) like where minor community reside?')
		input(' Press enter to proceed to plot')
		support_vector_machines()

	elif choice == 5:
		print(' Exiting......')
		loop = False # Exiting the loop

	else:
		input(' Wrong option selection. Enter any key to try again..')
