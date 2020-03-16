import think_paulo
import numpy as np
import matplotlib.pyplot as plt


lst = list(range(10))
hist = think_paulo.HistP(lst)

amostra = 2*np.random.randn(10000)+10


rank = think_paulo.percentile_rank(amostra, 10)


#think_paulo.probability_plot(amostra)


print(think_paulo.median(amostra))
mean = think_paulo.raw_moment(amostra,1)
variance = think_paulo.central_moment(amostra, 2)
skewness = think_paulo.skewness(amostra)
pearson_skewness = think_paulo.pearson_median_skewness(amostra)
cov = think_paulo.covariance(amostra,amostra)
pearson_corr = think_paulo.person_correlation(amostra, amostra)
spearman_corr = think_paulo.spearman_correlation(amostra, amostra)

print(f'Mean: {mean}')
print(f'variance: {variance}')
print(f'skewness: {skewness}')
print(f'pearson_skewness: {pearson_skewness}')
print(f'Cov: {cov}')
print(f'Pearson Correlation: {pearson_corr}')
print(f'Spearman_corr: {spearman_corr}')
think_paulo.estimate_1()
think_paulo.estimate2()

# sample_means = think_paulo.simulate_sample(mu=90, sigma=7.5, n=9, m=1000)
# print(f'Sample mean: {np.mean(sample_means)}')
# print(f'CI interval 5-95: {np.percentile(sample_means,5)} - {np.percentile(sample_means,95)}')
# print(f'Standard Error: {think_paulo.rmse(sample_means,90)}')

# dice_test = think_paulo.DiceTest([10,20,10,0,10,10])
# dice_test_P = dice_test.pvalue()
# print(f'P Value Dice Test: {dice_test_P}')

# dice_Chi_test = think_paulo.DiceChiTest([10,20,10,0,10,10])
# dice_Chi_test_P = dice_Chi_test.pvalue()
# print(f'P Value Dice Chi Test: {dice_Chi_test_P}')

import first

live, firsts, others = first.MakeFrames()
data = firsts.prglngth.values, others.prglngth.values
# ht = think_paulo.DiffMeansPermute(data)
# pvalue = ht.pvalue()
# print(f'PValue Mean Preg lenth diference: {pvalue}')

# ht = think_paulo.DiffMeansOneSided(data)
# pvalue = ht.pvalue()
# print(f'PValue Mean Preg lenth diference one sided: {pvalue}')

# cleaned = live.dropna(subset=['agepreg', 'totalwgt_lb'])
# data = cleaned.agepreg.values, cleaned.totalwgt_lb.values
# ht = think_paulo.CorrelatonPermute(data)
# pvalue = ht.pvalue()
# pvalue
# print(f'Correlation mother age and weight pvalue: {pvalue:5f}')

# data = firsts.prglngth.values, others.prglngth.values
# ht = think_paulo.PregLengthTest(data)
# p_value = ht.PValue()
# print('p-value =', p_value)
# print('actual =', ht.actual)
# print('ts max =', ht.MaxTestStat())


cleaned = live.dropna(subset=['agepreg', 'totalwgt_lb'])
inter, slope = think_paulo.least_squares(cleaned.agepreg.values, cleaned.totalwgt_lb.values)
fit_x, fit_y = think_paulo.fitLine(cleaned.agepreg.values,inter, slope)

print(f'{inter}, {slope}')
fig = plt.figure()
plt.plot(fit_x, fit_y)


live, firsts, others = first.MakeFrames()
live = live.dropna(subset=['agepreg', 'totalwgt_lb'])
ht = think_paulo.SlopeTest((live.agepreg, live.totalwgt_lb))
pvalue = ht.PValue()
print(f'Slope Test pvalue: {pvalue:5f}')


import brfss

df = brfss.ReadBrfss(nrows=None)
df = df.dropna(subset=['htm3', 'wtkg2'])
inter, slope = think_paulo.least_squares(df.htm3.values, np.log10(df.wtkg2).values)
print (f'Slope: {slope} inter: {inter}')
#ht = think_paulo.SlopeTest((df.htm3, np.log(df.wtkg2)))
#pvalue = ht.PValue()
#print(f'Slope Test pvalue: {pvalue:5f}')
#print(df)
a = 1

