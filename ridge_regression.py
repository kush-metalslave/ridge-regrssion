import numpy as np
#x=type some value as an numpy array
#y=type some value as an numpy array
#you can either give x and y values of your own or load them by using pandas.csv_load for loading a csv data
#demo:
x=np.array([[1,2,3,4],[5,2.3,9,3],[6.9,2,2,6.9],[1.1,2.2,3.6,4],[-3.2,5,9.08,12],[9.9,2.1,7.9,2.5],[6.78,2.12,8.9,4.5],[1.7,9.8,3.8,3.7],[6.1,2.34,7.9,2.8],[7.99,2.9,7.4,2.3],[6.8,11.9,7.1,5],[6.5,2,8,94.5]]).reshape((4,12))
y=np.array([24,77,177,27,80,62,41,13,73,75,97,102])
m=x.shape[0]
#m is the number of instances
#alpha=type some value
#lambda_val=type some value
#EPOCHS=type some value
#the above mentioned parameters are subject to the model, different combinations of these parameters should be tested to find the best possible combination
#demo:
alpha=0.2
lambda_val=0.01
epochs=10000
parameters=np.random.random_sample((4,1))#random initial parameters for ridge regression
parameters_normal=parameters#random initial parameters for normal regression(if  you want to compare both)
print(parameters)

#updating process
for i in range(EPOCHS):
    #calculating standard error
    standard_error=(1/m)*np.sum(np.power((np.dot(parameters_normal.T,x)-y),2))
    #calculating ridge_regression error
    ridge_regression_error=(1/m)*np.sum(np.power((np.dot(parameters.T,x)-y),2))+np.sum(np.power(parameters,2))
    #weight update for ridge
    parameters=parameters-((alpha*(1/m)*(np.sum(np.dot(parameters.T,x)-y)))+(lambda_val*parameters))
    #weight update for normal regression
    parameters_normal=parameters_normal-(alpha*(1/m)*(np.sum(np.dot(parameters.T,x)-y)))
    
#print("error after applying normal regression is :",standard_error)
print("error after applying ridge regression is :",ridge_regression_error)
print("final weights for ridge regression is :",parameters)
