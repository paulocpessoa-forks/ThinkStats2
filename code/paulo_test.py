import think_paulo
import numpy as np

lst = list(range(10))
hist = think_paulo.HistP(lst)

amostra = np.random.randn(10000)+10


rank = think_paulo.percentile_rank(amostra, 10)

a = 1
