from sklearn.utils import check_consistent_length
#import numpy as np #dependencies
#note: please change your classes to 1,2,...,n before using this function
#coded by: Quoc-Thang Phan and Chanh-Nghiem Nguyen
def di_scoring(y_true, y_score,accu):
    check_consistent_length(y_true, y_score)
    y_true_=np.copy(y_true) #duplicates original list for later uses
    y_true=np.array(y_true) #converts list to numpy array for subtraction with 1
    y_true=y_true-1 #y_true-1 converts grade [1,2,3,n] to column indices [0,1,2,n]
    y_true=y_true.tolist() #converts back to list
    flat=np.copy(y_score) #flat = probability (duplication using np.copy to prevent pointer behaviours of .sort())
    flat.sort() #flat is a list of sorted elements, from smallest to largest
    cndts= np.argmax(y_score,axis=1)+1 #max indices of grade probability array + 1 (due to 0 indices)
    cndts=cndts==y_true_ #note: conditions = y_true_ is the true grade, cndts is true if y_pred (max)== y_true 
    # np.in1d(cndts, y_true_) #optional, same as the line above
    inds = np.arange(len(y_true_))[cndts] #indices such that y_pred (max prob) == y_true 
    adm=np.mean(abs(np.amax(y_score[inds],axis=1)-flat[inds][:,-2])) #max - second largest element
    aem=y_score[range(len(y_true)),y_true]- np.amax(y_score,axis=1) #error= y_score_true - max, only misclassified > 0
    aem=np.true_divide(abs(aem).sum(),abs(aem!=0).sum()) #take the average of error to get aem
    if np.isnan(aem) == True: 
        aem=0
    di=(adm*accu)-(aem*(1-accu))
    return adm,aem,di


#example
#from sklearn.metrics import accuracy_score #calculate accuracy of truth vs prediction.

#train_adm,train_aem,train_di = di_scoring(y_train,y_trainprobability,accuracy_score(y_train,y_fit))
#test_adm,test_aem,test_di = di_scoring(y_test,y_testprobability,accuracy_score(y_test,y_pred))

#print('train_adm,train_aem,tran_di',train_adm,train_aem,train_di)
#print('test_adm,test_aem,test_di',test_adm,test_aem,test_di)
