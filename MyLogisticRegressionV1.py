# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:27:58 2016
@author: lui_2
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:15:46 2016
@author: lui
"""
import functools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy  as np
from numpy import arange, ones, exp, log
from numpy.random import seed,random_sample,random

#--- Useful classes
class RecPlayData:
    ''' records some veriables at each step of a procedures and replayes them when necessaty '''
    _data_bag = {}
    
    def rec(self,step,**argv):
        self._data_bag[step] = argv
    
    @property 
    def len(self):
        return len(self._data_bag)
    
    def __getitem__(self, step, key=None):
            return self._data_bag[step]

    def __iter__(self):
        for key,item in self._data_bag.items():
            yield key,item
    

#-------------- Utils functions


def ifdebug_lvl(lvl):
    ''' This decorator excutes a functions only if the required debug_level is 
    in the list of the levels passed in'''
    def actualDecorator(test_func):
        @functools.wraps(test_func)       
        def inner(*args,**argv):
            if(lvl in debug_lvls):
                return test_func(*args,**argv)
        return inner
    return actualDecorator
    
#---------------- Application functions

# debug levels
debug_lvls = {}



def misclassified(prediction_prob):
    prediction =(prediction_prob >= 0.5).round()
    error = np.abs(classes-prediction).sum()
    return error

# print the points separated by the two classes and the function itself
def plot_graph(classes,w,title):
    plt.clf()
    plt.scatter(x[classes==True],y[classes==True],s=20,c='g')
    plt.scatter(x[classes==False],y[classes==False],s=20,c='r')
    plt.axis([-50, 50, -50, 50])
    plt.plot( x,POLY(x) )
    decision_boundary = create_decision_boundary(w)
    red_patch = mpatches.Patch(color='red', label='class 0')
    green_patch = mpatches.Patch(color='green', label='class 1')
    plt.legend(handles=[red_patch,green_patch])
    plt.title(title)
    ln, = plt.plot(x,decision_boundary(x))

def create_decision_boundary(w):
    '''Given the weights as coefficients of the equation of a line generates a
    function and returns it'''
    return lambda x: (-w[2]*x-w[0])/w[1]

@ifdebug_lvl(9)
def debug_info(*args,**argv):
    print(misclassified(prediction_prob))
    print( "cost = ",bce(w,training_set,classes*1) )
    #print(w)

@ifdebug_lvl(8)    
def plot_n_save(i):    
    ''' plot the graph and save the picture on a file '''
    plt.figure(i)
    plot_graph(classes,w)
    plt.savefig('lr'+str(i)+'.png')
    
                      
#################################





# we want to be able to reproduce the the creation of the training set
seed(7)

# number of training example
SAMPLE_NUMBER = 100
LEARNING_RATE = 0.01
EPOCHS = 11
# create a POLYnomial function to provide a complex boundary
POLY = lambda x: -0.0000000133*(x-50)*(x+2)*(x-10)*(x-20)*(x+40)**2-1


# we initialize the weights to random values
w = random(3)
y = (random_sample(SAMPLE_NUMBER)-0.5) * 50
x = arange(-50.,50.,1)



# The divider for the classes is the above POLYnomial
# Boolean are interchangable with 0 and 1 so I wont converrt them but if you like
# you can add .astype('int') at the end of the first line
classes  = y>POLY(x)

# prepare the sigmoid and its derivative
sig = lambda x: 1/(1+exp(-x))
sig_der = lambda x: sig(x)*(1-sig(x))
bce = lambda w,x,y: (-y*log(sig(x.T @ w))+(1-y)*log(1-sig(x.T @ w))).sum()/len(x)


#package x and y into one array, we need to add the bias term as well 
training_set = np.vstack( (ones(SAMPLE_NUMBER),y, x) )

record_w = RecPlayData()
for i in range(EPOCHS):
    prediction_prob = sig( training_set.T @ w )                                # (3x100).T = 100x3 * 3x1 = 100x1
    w_update = ( training_set @ (prediction_prob - classes) ) / SAMPLE_NUMBER  #  3x100 * 100x1 = 3*1
    w  -= LEARNING_RATE * w_update
    
    error = misclassified(sig( training_set.T @ w ))
    record_w.rec(i,w=w.copy(),err=error)
    #debug_info()
    #plot_n_save(i)
    

# Generate a graph for each set of weights

for k,item in record_w:
    title = "step {} - acc = {} - mis={}".format(k,np.round(1-item['err']/100,2),item['err'])   
    plt.figure(k)
    plot_graph(classes,item['w'],title)
    
prediction_prob = sig( training_set.T @ w )
decision_boundary = create_decision_boundary(w)
# Check that the decision boundary we have plotted really has that number of misclassification to which the algorith converged
pred_pos =  y>=decision_boundary(x)
pred_neg =  y<decision_boundary(x)

# once we have the decision boundary we want to know how many training examples have been misclassified
# we sum those above the line (which should be all 1s ) that are 0s with those below the line that should be 0s 
# but are 1s
bad_pred = (classes[pred_pos]==0).sum() + (classes[pred_neg]==1).sum()
print("# of bad predictions: ",bad_pred)
print( misclassified( sig( training_set.T @ w ) ) )



# Questions
# 1 why it does not converge to the minimum
# why my update is a sign different from the lesson in 
#print(np.log((-y+1)/y))



    

