# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:12:08 2021

@author: abhin
"""
import numpy as np
import matplotlib.pyplot as plt

n=100
c1,c2,c3=0,25,50
m=50
alphas=np.linspace(0,2000,2000)
res1,res2,res3=[],[],[]

for alpha in alphas:
    res1.append((alpha+c1)/((m*alpha)+n))
    res2.append((alpha+c2)/((m*alpha)+n))
    res3.append((alpha+c3)/((m*alpha)+n))
plt.figure(figsize=[10,5])
plt.plot(alphas, res1, label="c = {}".format(c1))
plt.plot(alphas, res2, label="c = {}".format(c2))
plt.plot(alphas, res3, label="c = {}".format(c3))
plt.ylabel("Predictive Word Probability")
plt.xlabel("Alpha")
plt.title("Predictive Word Probabilities For Varying Alpha With n = 100, M = 50")
plt.legend()
plt.show()