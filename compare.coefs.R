
bugs.coefs <- read.table("radon.coefs.from.bugs.csv",header=FALSE)
pymc.coefs <- read.table("radon.coefs.from.pymc.csv",header=FALSE,sep=",")

coef.diffs <- (bugs.coefs - pymc.coefs) / (bugs.coefs + pymc.coefs)
print(round(coef.diffs,2))
