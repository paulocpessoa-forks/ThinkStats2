from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math



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

	Pearson’s correlation is always between -1 and +1 (including both). If ρ is
	positive, we say that the correlation is positive, which means that when one
	variable is high, the other tends to be high. If ρ is negative, the correlation
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



