# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:43:10 2021

@author: abhin
"""

from sampleDiscrete import sampleDiscrete
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(1)
# load data
data = sio.loadmat('kos_doc_data.mat')
A = np.array(data['A'])
B = data['B']
V = data['V']
m=np.size(V)
n=0
totp=0
alpha=9
wordsa=defaultdict(int)
for w in A:
    wordsa[w[1]]+=w[2]
    n+=w[2]
betas=[0 for i in range(m)]
print(wordsa.keys())
for i in range(m):
    betas[i]=(wordsa[i+1]+alpha)/((m*alpha)+n)
probsb=defaultdict(int)
perp=defaultdict(int)
for doc_id,word_id,count in B:
    if word_id not in wordsa:
        print("Not IN, ", word_id)
    probsb[doc_id]+=(count*np.log(betas[word_id-1]))
    perp[doc_id]+=count
for doc in perp:
    perp[doc]=np.exp(-probsb[doc]/perp[doc])
print(probsb[2001],perp[2001])

plt.figure(figsize=[10,5])
plt.plot(list(perp.keys()), list(perp.values()), color="r")
plt.ylabel("Per-Word Perplexity")
plt.xlabel("Document")
plt.title("Per-Word Perplexity For Different Documents")
plt.show()

plt.figure(figsize=[10,5])
plt.plot(list(probsb.keys()), list(probsb.values()), color="c")
plt.ylabel("Log Probability")
plt.xlabel("Document")
plt.title("Log Probability For Different Documents")
plt.show()