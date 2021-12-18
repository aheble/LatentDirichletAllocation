# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:27:15 2021

@author: abhin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:48:29 2021

@author: abhin
"""

import scipy.io as sio
import numpy as np
from scipy.sparse import coo_matrix as sparse
from sampleDiscrete import sampleDiscrete
import matplotlib.pyplot as plt

def LDA(A, B, K, alpha, gamma):
    print("This is working")
    """
    Latent Dirichlet Allocation

    :param A: Training data [D, 3]
    :param B: Test Data [D, 3]
    :param K: number of mixture components
    :param alpha: parameter of the Dirichlet over mixture components
    :param gamma: parameter of the Dirichlet over words
    :return: perplexity, multinomial over words
    """
    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])])  # total number of unique words
    D = np.max(A[:, 0])  # number of documents in A

    # A's columns are doc_id, word_id, count
    swd = sparse((A[:, 2], (A[:, 1]-1, A[:, 0]-1))).tocsr()
    
    # Initialization
    skd = np.zeros((K, D))  # count of word assignments to topics for document d
    swk = np.zeros((W, K))  # unique word topic assignment counts across all documents

    s = []  # each element of the list corresponds to a document
    r = 0
    for d in range(D):  # iterate over the documents
        z = np.zeros((W, K))  # unique word topic assignment counts for doc d
        words_in_doc_d = A[np.where(A[:, 0] == d+1), 1][0]-1
        for w in words_in_doc_d:  # loop over the unique words in doc d
            c = swd[w, d]  # number of occurrences for doc d
            for i in range(c):  # assign each occurrence of word w to a doc at random
                k = np.floor(K*np.random.rand())
                z[w, int(k)] += 1
                r += 1
        skd[:, d] = np.sum(z, axis=0)  # number of words in doc d assigned to each topic
        swk += z  # unique word topic assignment counts across all documents
        s.append(sparse(z))  # sparse representation: z contains many zero entries

    sk = np.sum(skd, axis=1)  # word to topic assignment counts accross all documents
    # This makes a number of Gibbs sampling sweeps through all docs and words, it may take a bit to run
    num_gibbs_iters = 50
    mix_prop = np.zeros((num_gibbs_iters, K))
    for itera in range(num_gibbs_iters):
        print("We are on iteration: ", itera)
        for d in range(D):
            z = s[d].todense()  # unique word topic assigmnet counts for document d
            words_in_doc_d = A[np.where(A[:, 0] == d + 1), 1][0] - 1
            for w in words_in_doc_d:  # loop over unique words in doc d
                a = z[w, :].copy()  # number of times word w is assigned to each topic in doc d
                indices = np.where(a > 0)[1]  # topics with non-zero word counts for word w in doc d
                np.random.shuffle(indices)
                for k in indices:  # loop over topics in permuted order
                    k = int(k)
                    for i in range(int(a[0, k])):  # loop over counts for topic k
                        z[w, k] -= 1  # remove word from count matrices
                        swk[w, k] -= 1
                        sk[k] -= 1
                        skd[k, d] -= 1
                        b = (alpha + skd[:, d]) * (gamma + swk[w, :]) \
                            / (W * gamma + sk)
                        kk = sampleDiscrete(b, np.random.rand())  # Gibbs sample new topic assignment
                        z[w, kk] += 1  # add word with new topic to count matrices
                        swk[w, kk] += 1
                        sk[kk] += 1
                        skd[kk, d] += 1
        
            s[d] = sparse(z)  # store back into sparse structure
        stest=np.sum(skd, axis=1)
        for m in range(K):
            theta_k = (alpha+stest[m])/((alpha*K)+np.sum(stest))
            mix_prop[itera,m]=theta_k
            
    return mix_prop


if __name__ == '__main__':
    np.random.seed(4)
    # load data
    data = sio.loadmat('kos_doc_data.mat')
    A = np.array(data['A'])
    B = data['B']
    V = data['V']

    K = 20  # number of clusters
    alpha = .1  # parameter of the Dirichlet over mixture components
    gamma = .1  # parameter of the Dirichlet over words

    mix_prop = LDA(A, B, K, alpha, gamma)
    xaxis = [x+1 for x in range(50)]
    
    plt.figure(figsize=[10,5])
    for i in range(K):
        plt.plot(xaxis, mix_prop[:,i])
    plt.ylabel("Topic Posterior")
    plt.xlabel("Iteration")
    plt.xlim(left=1)
    plt.title("Seed = 4")
    plt.show()