from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class HistP():
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


def percentile(scores, p_rank):
	scores.sort()
	for score in scores:
		if percentile_rank(scores, score) >= p_rank:
			return score


def percentile2(scores, p_rank):
	scores.sort()
	index = p_rank * (len(scores) - 1) // 100
	return scores[index]
