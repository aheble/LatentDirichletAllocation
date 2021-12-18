# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:53:28 2021

@author: abhin
"""

from sampleDiscrete import sampleDiscrete
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from bmm import BMM

np.random.seed(1)
# load data
data = sio.loadmat('kos_doc_data.mat')
A = np.array(data['A'])
B = data['B']
V = data['V']
m=np.size(V)

perp, swk, mix_prop = BMM(A,B,m,    )

plt.figure(figsize=[10,5])
xaxis = [x+1 for x in range(num_iters_gibbs)]
for i in range(K):
    plt.plot(xaxis, mix_prop[:,i], color="r")
plt.ylabel("Mixing Proportion")
plt.xlabel("Iteration")
plt.title("Mixing Proportions Against Gibbs Sampling Iteration")
plt.show()
