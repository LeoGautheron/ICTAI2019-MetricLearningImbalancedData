#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import optimize


class iml():
    def __init__(self, pClass, k=1, a=1, b=1, m=1, Lambda=1,
                 randomState=np.random):
        self.pClass = pClass  # minority<=>positive class
        self.a = a
        self.b = b
        self.k = k
        self.m = m
        self.Lambda = Lambda
        self.randomState = randomState

    def fit(self, X, Y):
        self.X = X

        self.idxP = np.where(Y == self.pClass)[0]  # indexes of pos examples
        self.idxN = np.where(Y != self.pClass)[0]  # indexes of other examples
        self.Np = len(self.idxP)
        self.Nn = len(self.idxN)

        if self.Np <= 1:
            print("Error, there should be at least 2 positive examples")
            return

        # Initialize the number of neighbors
        if self.k >= self.Np:
            self.k = self.Np - 1  # maximum possible number of neighbors
        if self.k <= 0:
            self.k = 1  # we need at least one neighbor

        # Positive Positive Pairs
        D = euclidean_distances(self.X[self.idxP], squared=True)
        np.fill_diagonal(D, np.inf)
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        self.SimP_i = []
        self.SimP_j = []
        for idxI in range(len(self.idxP)):  # for each positive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.SimP_i.append(self.idxP[idxI])
                self.SimP_j.append(self.idxP[idxJ])
                idxIdxJ += 1
        self.SimP_i = np.array(self.SimP_i)
        self.SimP_j = np.array(self.SimP_j)

        D = euclidean_distances(self.X[self.idxP], self.X[self.idxN],
                                squared=True)

        # Positive Negative Pairs
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        self.DisP_i = []
        self.DisP_j = []
        for idxI in range(len(self.idxP)):  # for each positive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.DisP_i.append(self.idxP[idxI])
                self.DisP_j.append(self.idxN[idxJ])
                idxIdxJ += 1
        self.DisP_i = np.array(self.DisP_i)
        self.DisP_j = np.array(self.DisP_j)

        # Negative Positive Pairs
        Didx = np.argsort(D.T)  # indexes for matrix D.T sorted ascending
        self.DisN_i = []
        self.DisN_j = []
        for idxI in range(len(self.idxN)):  # for each negative example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.DisN_i.append(self.idxN[idxI])
                self.DisN_j.append(self.idxP[idxJ])
                idxIdxJ += 1
        self.DisN_i = np.array(self.DisN_i)
        self.DisN_j = np.array(self.DisN_j)

        # Negative Negative Pairs
        D = euclidean_distances(self.X[self.idxN], squared=True)
        np.fill_diagonal(D, np.inf)
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        self.SimN_i = []
        self.SimN_j = []
        for idxI in range(len(self.idxN)):  # for each negative example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.SimN_i.append(self.idxN[idxI])
                self.SimN_j.append(self.idxN[idxJ])
                idxIdxJ += 1
        self.SimN_i = np.array(self.SimN_i)
        self.SimN_j = np.array(self.SimN_j)

        # Call the L-BFGS-B optimizer with the identity matrix as initial point
        L, loss, details = optimize.fmin_l_bfgs_b(
                       maxiter=200, func=self.loss_grad, x0=np.eye(X.shape[1]))

        # Reshape result from optimizer
        self.L_ = L.reshape(X.shape[1], X.shape[1])

    def loss_grad(self, L):
        L = L.reshape((self.X.shape[1], self.X.shape[1]))
        M = L.T.dot(L)

        # Compute pairwise mahalanobis distance between the examples
        # with the current projection matrix L
        Dm_pp = np.sum((self.X[self.SimP_i].dot(L.T) -
                        self.X[self.SimP_j].dot(L.T))**2, axis=1)
        Dm_pn = np.sum((self.X[self.DisP_i].dot(L.T) -
                        self.X[self.DisP_j].dot(L.T))**2, axis=1)
        Dm_np = np.sum((self.X[self.DisN_i].dot(L.T) -
                        self.X[self.DisN_j].dot(L.T))**2, axis=1)
        Dm_nn = np.sum((self.X[self.SimN_i].dot(L.T) -
                        self.X[self.SimN_j].dot(L.T))**2, axis=1)

        # Sim+ (Positive, Positive) pairs
        idx = np.where(Dm_pp > 1)[0]
        diff = self.X[self.SimP_i[idx]] - self.X[self.SimP_j[idx]]
        SimP_g = 2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        SimP_l = np.sum(Dm_pp[idx]) - len(idx)  # loss

        # Dis+ (Positive, Negative) pairs
        idx = np.where(Dm_pn < 1 + self.m)[0]
        diff = self.X[self.DisP_i[idx]] - self.X[self.DisP_j[idx]]
        DisP_g = -2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        DisP_l = len(idx)*(1 + self.m) - np.sum(Dm_pn[idx])  # loss

        # Dis- (Negative, Positive) pairs
        idx = np.where(Dm_np < 1 + self.m)[0]
        diff = self.X[self.DisN_i[idx]] - self.X[self.DisN_j[idx]]
        DisN_g = -2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        DisN_l = len(idx)*(1 + self.m) - np.sum(Dm_np[idx])  # loss

        # Sim- (Negative, Negative) pairs
        idx = np.where(Dm_nn > 1)[0]
        diff = self.X[self.SimN_i[idx]] - self.X[self.SimN_j[idx]]
        SimN_g = 2*L.dot(diff.T.dot(diff))  # gradient (sum of outer products)
        SimN_l = np.sum(Dm_nn[idx]) - len(idx)  # loss

        # Squared Frobenius norm term
        identity = np.eye(M.shape[0])
        N_g = 4*L.dot(L.T.dot(L) - identity)  # gradient
        N_l = np.sum((M-identity)**2)  # loss

        loss = (self.a*SimP_l +
                self.b*DisP_l +
                (1-self.b)*DisN_l +
                (1-self.a)*SimN_l +
                self.Lambda*N_l)
        gradient = (self.a*SimP_g +
                    self.b*DisP_g +
                    (1-self.b)*DisN_g +
                    (1-self.a)*SimN_g +
                    self.Lambda*N_g)

        return loss, gradient.flatten()

    def transform(self, X):
        return X.dot(self.L_.T)
