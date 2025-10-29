from scipy.stats import chi2
import numpy as np
alpha = 0.995
trust_range = np.sqrt(chi2.ppf(alpha, df=2))  # 2.448
print(trust_range)
