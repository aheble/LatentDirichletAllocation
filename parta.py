# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:08:07 2021

@author: abhin
"""

from sampleDiscrete import sampleDiscrete
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
# load data
data = sio.loadmat('kos_doc_data.mat')
A = np.array(data['A'])
B = data['B']
V = data['V']
num_words=np.size(V)
word_count=np.zeros(num_words)
total_count=0
for i in A:
    total_count+=i[2]
    word_count[i[1]-1]+=i[2]
beta=list((word_count[x]/total_count,x) for x in range(num_words))
beta.sort(reverse=True,key=lambda x:x[0])
top_words=[]
freq_words=[]
for i,w in enumerate(beta[:20]):
    #print(V[w[1]][0][0])
    top_words.append(V[w[1]][0][0])
    freq_words.append(w[0])
print(beta[0])
plt.figure(figsize=[10,5])
plt.barh(top_words[::-1],freq_words[::-1],height=0.8)
plt.ylabel("Word")
plt.xlabel("Probability")
plt.title("The 20 Greatest Probability Words")
plt.show()
