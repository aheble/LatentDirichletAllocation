# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:53:57 2021

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
n=np.size(V)
totw=0
wordsa=defaultdict(int)
for w in A:
    wordsa[w[1]]+=w[2]
    totw+=w[2]
m=len(wordsa.keys())
wordsb=defaultdict(int)
totb=0
for x in B:
    doc_id,word_id,count=x
    if doc_id==2001:
        wordsb[word_id]+=count
        totb+=count
alphas=np.linspace(0,500,500)
logprob=[]
perp=[]
for alpha in alphas:
    totp=0
    for w in wordsb.keys():
        totp+=wordsb[w]*np.log((alpha+wordsa[w])/((n*alpha)+totw))
    logprob.append(totp)
    perp.append(np.exp(-totp/totb))

minperp=min(perp)
minalpha=alphas[perp.index(minperp)]
print(minalpha)

plt.figure(figsize=[10,5])
plt.plot(alphas, logprob, color="c")
#plt.plot(alphas, perp, label="perp")
plt.ylabel("Log Probability")
plt.xlabel("Alpha")
plt.title("Log Probability For Document ID 2001 With Varying Alpha")
plt.show()

plt.figure(figsize=[10,5])
plt.plot(alphas, perp, color="r")
plt.ylabel("Per-Word Perplexity")
plt.xlabel("Alpha")
plt.title("Per-Word Perplexity For Document ID 2001 With Varying Alpha")
plt.show()