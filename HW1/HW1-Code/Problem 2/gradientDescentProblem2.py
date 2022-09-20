# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 22:15:13 2021

@author: Alireza Khayyatian
"""
import numpy as np
''' Define gradient descent function  ''' 

def gradientDescent(x, y,x_test,y_test, theta_MSE, alpha, m,s, numIterations
                    ,error_train_MSE
                    ,error_test_MSE 
                    ,gradient_MSE_set):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        
        #print(i)
        hypothesis_MSE = np.dot(x, theta_MSE)
        
        loss_MSE = hypothesis_MSE - y

        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        
        cost_MSE = np.sum(loss_MSE ** 2) / (2 * m) #MSE
        
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient_MSE = np.dot(xTrans, loss_MSE) / m
        # update
        theta_MSE = theta_MSE - alpha * gradient_MSE
        
        error_train_MSE.append(cost_MSE)
        
        gradient_MSE_set.append(gradient_MSE)
        
        #calculate Test Error    
        hypothesis_test_MSE = np.dot(x_test, theta_MSE)

        loss_test_MSE = hypothesis_test_MSE - y_test

        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        
        cost_test_MSE = np.sum(loss_test_MSE ** 2) / (2 * s) #MSE

        error_test_MSE.append(cost_test_MSE)

    return (theta_MSE)


