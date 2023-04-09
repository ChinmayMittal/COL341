from best import *

print("Testing binary SVM.....")

T = best_classifier_two_class()
# print("Initializing binary classifier")
try:
	T.fit("bi_train.csv")
	y_pred = T.predict("bi_test.csv")
	y_exp = [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
	acc = 0
	for i in range(39):
			if (y_exp[i] == y_pred[i]):
					acc+=1
	print('Accuracy of your best model on the test set is '+str(acc)+'/39')
except:
     print("There is a bug in your binary SVM code!")
print("")
print("Testing multiclass SVM.....")

T = best_classifier_multi_class()
try:
	print("Initializing multi classifier.....")
	T._init_trainers()
	T.fit("multi_train.csv")
	y_pred = T.predict("multi_test.csv")
	y_exp = [9, 4, 3, 5, 4, 3, 9, 4, 7, 6, 3, 4, 8, 7, 5, 7, 8, 5, 9, 6, 2, 7, 6, 1, 5, 1, 3, 6, 6, 4, 2, 6, 6, 3, 3, 4, 3, 7, 7]
	acc = 0
	for i in range(39):
			if (y_exp[i] == y_pred[i]):
					acc+=1
	print('Accuracy of your best model on the multi test set is '+str(acc)+'/39')
except:
     print("There is a bug in your multi SVM code!")
print("")
print("Testing done..... exiting")