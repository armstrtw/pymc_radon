import csv
import numpy as np
import pymc
from math import log

radon_csv = csv.reader(open('srrs.csv'))
radon = []
for row in radon_csv:
    radon.append(tuple(row))

counties = np.array([x[0] for x in radon])
y = np.array([float(x[1]) for x in radon])
x = np.array([float(x[2]) for x in radon])

## gelman adjustment for log
y[y==0]=.1
y = np.log(y)

## groupings
def createCountyIndex(counties):
    counties_uniq = sorted(set(counties))
    counties_dict = dict()
    for i, v in enumerate(counties_uniq):
        counties_dict[v] = i
    ans = np.empty(len(counties),dtype='int')
    for i in range(0,len(counties)):
        ans[i] = counties_dict[counties[i]]
    return ans

index_c = createCountyIndex(counties)

# Priors
mu_b = pymc.Normal('mu_b', mu=0., tau=0.0001)
sigma_b = pymc.Uniform('sigma_b', lower=0, upper=100)
tau_b = pymc.Lambda('tau_b', lambda s=sigma_b: s**-2)

a = pymc.Normal('a', mu=0., tau=0.0001)
b = pymc.Normal('b', mu=mu_b, tau=tau_b, value=np.zeros(len(set(counties))))

sigma_y = pymc.Uniform('sigma_y', lower=0, upper=100)
tau_y = pymc.Lambda('tau_y', lambda s=sigma_y: s**-2)

# Model
@pymc.deterministic(plot=False)
def y_hat(a=a,b=b):
       return a + b[index_c]*x

# Likelihood
@pymc.stochastic(observed=True)
def y_i(value=y, mu=y_hat, tau=tau_y):
    return pymc.normal_like(value,mu,tau)
