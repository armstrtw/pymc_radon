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

J = len(set(counties))
K = 2
df = K + 1

## use matrix form
X = np.empty((len(y),K))
X[:,0] = 1.
X[:,1] = x

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
sigma_y = pymc.Uniform('sigma_y', lower=0, upper=100)
tau_y = pymc.Lambda('tau_y', lambda s=sigma_y: s**-2)

xi = pymc.Uniform('xi', lower=0, upper=100, value=np.zeros(K))
mu_raw = pymc.Normal('mu_raw', mu=0., tau=0.0001,value=np.zeros(K))
Tau_B_raw = pymc.Wishart('Tau_B_raw', df, Tau=np.diag(np.ones(K)))
B_raw = []
for i in range(J):
    B_raw.append(pymc.MvNormal('B_raw_%i' % i, mu_raw, Tau_B_raw))

@pymc.deterministic
def B_raw_m(B_raw=B_raw):
    return np.vstack(B_raw)

@pymc.deterministic
def Sigma_B_raw(Tau_B_raw=Tau_B_raw):
    return np.linalg.inv(Tau_B_raw)

@pymc.deterministic
def rho_B(Sigma_B_raw=Sigma_B_raw):
    return Sigma_B_raw / np.sqrt(np.diag(Sigma_B_raw) * Sigma_B_raw)

@pymc.deterministic
def Sigma_B(xi=xi,Sigma_B_raw=Sigma_B_raw):
    return abs(xi) * np.sqrt(np.diag(Sigma_B_raw))

@pymc.deterministic(plot=True)
def B(xi=xi, B_raw_m=B_raw_m):
    return xi * B_raw_m

@pymc.deterministic
def mu(xi=xi, mu_raw=mu_raw):
    return xi * mu_raw

# Model
@pymc.deterministic(plot=False)
def y_hat(B=B):
       return (B[index_c,] *X).sum(axis=1)

# Likelihood
@pymc.stochastic(observed=True)
def y_i(value=y, mu=y_hat, tau=tau_y):
    return pymc.normal_like(value,mu,tau)
