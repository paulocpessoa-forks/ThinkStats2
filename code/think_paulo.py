from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


class UnimplementedMethodException(Exception):
    pass

class HistP:
    """
    Uma classe que contém um dicionário
    """

    def __init__(self, obj):
        self.d = {}

        if obj is None:
            return

        if isinstance(obj, pd.Series):
            self.d.update(obj.value_counts().iteritems())

        else:  # Trata como uma lista
            self.d = {}
            for x in obj:
                self.d[x] = self.d.get(x, 0) + 1
        # poderia ser apenas self.d = Counter(obj)


def percentile_rank(scores, your_score):
    count = 0
    for score in scores:
        if score <= your_score:
            count += 1
    p_rank = 100.0 * count / len(scores)
    return p_rank


def percentile (scores, p_rank):
    scores.sort()
    for score in scores:
        if percentile_rank(scores, score) >= p_rank:
            return score


def percentile2(scores, p_rank):
    scores.sort()
    index = p_rank * (len(scores) - 1) // 100
    return scores[index]


def probability_plot(amostra):
    normal_dist = np.random.randn(len(amostra))
    normal_dist = np.sort(normal_dist)
    amostra = np.sort(amostra)
    plt.plot(normal_dist, amostra)


def raw_moment(amostra, k):
    """
    A statistic based on the sum of data raised to a power.
    Ex: raw_moment(amostra, 1) é igual a mean
    """
    return sum(amostra**k)/len(amostra)


def central_moment(amostra, k):
    """
     A statistic based on deviation from the mean, raised
    to a power.
    Ex: central_moment(amostra, 2) é igual a variância
    """
    mean = raw_moment(amostra, 1)
    return sum((amostra-mean)**k) / len(amostra)


def median(amostra):
    return percentile2(amostra, 50)


def skewness(amostra):
    """
     A measure of how asymmetric a distribution is
    """
    var = central_moment(amostra, 2)
    std = np.sqrt(var)
    return central_moment(amostra, 3) / std**3


def pearson_median_skewness(amostra):
    """
    A statistic intended to quantify
    the skewness of a distribution based on the median, mean, and standard
    deviation.
    """
    med = median(amostra)
    mean = raw_moment(amostra, 1)
    var = central_moment(amostra, 2)
    std = math.sqrt(var)
    gp = 3 * (mean - med) / std
    return gp


def covariance(amostra_1, amostra_2):
    mean_1 = raw_moment(amostra_1, 1)
    mean_2 = raw_moment(amostra_2, 1)
    cov = ((mean_1 - amostra_1) @ (mean_2 - amostra_2)) / len(amostra_1)
    return cov


def person_correlation(amostra_1, amostra_2):
    """

    @param amostra_1:
    @param amostra_2:
    @return: pearson_corr

    Pearson’s correlation is always between -1 and +1 (including both). If ϝ is
    positive, we say that the correlation is positive, which means that when one
    variable is high, the other tends to be high. If ϝ is negative, the correlation
    is negative, so when one variable is high, the other is low.
    """
    mean_1 = raw_moment(amostra_1, 1)
    std_1 = math.sqrt(central_moment(amostra_1, 2))
    delta_1 = (mean_1 - amostra_1) / std_1
    mean_2 = raw_moment(amostra_2, 1)
    std_2 = math.sqrt(central_moment(amostra_2, 2))
    delta_2 = (mean_2 - amostra_2) / std_2
    pearson_corr = (delta_1 @ delta_2) / len(amostra_1)
    return pearson_corr


def spearman_correlation(amostra_1, amostra_2):
    """

    @param amostra_1:
    @param amostra_2:
    @return: spearman_corr

    Spearman’s rank correlation is an alternative that mitigates
    the effect of outliers and skewed distributions. To compute Spearman’s correlation, we have to compute the rank of each value, which is its index in
    the sorted sample.
    """
    a_1_rank = pd.Series(amostra_1).rank()
    a_2_rank = pd.Series(amostra_1).rank()
    return person_correlation(a_1_rank,a_2_rank)


def rmse(estimates,actual):
    e2 = [(estimate-actual)**2 for estimate in estimates]
    mse = np.sum(e2)/len(estimates)
    return np.sqrt(mse)


def estimate_1(n=7, m=1000):
    mu = 0
    sigma = 1
    means = []
    medians = []
    for _ in range(m):
        amostra = np.random.normal(mu, sigma, n)
        means.append(np.mean(amostra))
        medians.append(np.median(amostra))

    print(f'Mean RMSE: {rmse(means, mu)}')
    print(f'Median RMSE: {rmse(medians, mu)}')


def estimate2(n=7, m=1000):
    mu = 0
    sigma = 1
    estimates1 = []
    estimates2 = []
    for _ in range(m):
        xs = [random.gauss(mu, sigma) for i in range(n)]
        biased = np.var(xs)
        unbiased = np.var(xs, ddof=1)
        estimates1.append(biased)
        estimates2.append(unbiased)
    print('mean error biased', mean_error(estimates1, sigma ** 2))
    print('mean error unbiased', mean_error(estimates2, sigma ** 2))


def mean_error(estimates, actual):
    errors = [estimate-actual for estimate in estimates]
    return np.mean(errors)


def simulate_sample(mu=90, sigma=7.5, n=9, m=1000):
    means = []
    for _ in range(m):
        amostra = np.random.normal(mu, sigma, n)
        means.append(np.mean(amostra))
    return means


class HypotesisTest(object):

    def __init__(self, data):
        self.data = data
        self.make_model()
        self.actual = self.test_statistic(data)

    def pvalue(self,iters=1000):
        self.test_stats = [self.test_statistic(self.run_model())
                           for _ in range(iters)]
        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def test_statistic(self, data):
        raise UnimplementedMethodException()

    def make_model(self):
        pass

    def run_model(self):
        raise UnimplementedMethodException()


class CoinTest(HypotesisTest):

    def test_statistic(self, data):
        heads, tails = data
        test_stat = abs(heads - tails)
        return test_stat

    def run_model(self):
        heads, tails = self.data
        n = heads + tails
        sample = [random.choice('HT') for _ in range(n)]
        hist = HistP(sample).d
        data = hist['H'], hist['T']
        return data


class DiffMeansPermute(HypotesisTest):

    def test_statistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def make_model(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1,group1))

    def run_model(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data


class DiffMeansOneSided(DiffMeansPermute):

    def test_statistic(self, data):
        group1, group2 = data
        test_stat = group1.mean() - group2.mean()
        return test_stat


class CorrelatonPermute(HypotesisTest):

    def test_statistic(self, data):
        xs, ys = data
        test_stat = person_correlation(xs, ys)
        return test_stat

    def run_model(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys


class DiceTest(HypotesisTest):

    def test_statistic(self, data):
        observed = data
        n = np.sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = np.sum(abs(observed-expected))
        return test_stat

    def run_model(self):
        n = np.sum(self.data)
        values = [1, 2, 3, 4, 5, 6]
        choices = [np.random.choice(values) for _ in range(n)]
        hist = HistP(choices).d
        xs = np.random.permutation(xs)
        return xs, ys

