import think_paulo
import numpy as np

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

sample_means = think_paulo.simulate_sample(mu=90, sigma=7.5, n=9, m=1000)
print(f'Sample mean: {np.mean(sample_means)}')
print(f'CI interval 5-95: {np.percentile(sample_means,5)} - {np.percentile(sample_means,95)}')
print(f'Standard Error: {think_paulo.rmse(sample_means,90)}')

a = 1

