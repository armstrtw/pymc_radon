library (MCMCpack)
library(R2jags)

srrs <- read.csv("srrs.csv",header=FALSE,as.is=TRUE)
srrs <- srrs[order(srrs[,1]),]
n <- nrow(srrs)
county <- as.integer(as.factor(srrs[,1]))
J <- length(unique(county))
X <- cbind(rep(1,nrow(srrs)),srrs[,3])
radon <- srrs[,2]
y <- log(ifelse (radon==0, .1, radon))
K <- ncol(X)
W <- diag(K)


model.names <- list ("n", "J","K","y","county","X","W")
parameters.to.save <- c("B","mu","sigma.y","sigma.B","rho.B","e.y")
model.inits <- function () {
    list(B.raw=array(rnorm(J*K), c(J,K)),
         mu.raw=rnorm(K),
         sigma.y=runif(1),
         Tau.B.raw=rwish(K+1,diag(K)),
         xi=runif(K))
}

wd <- getwd()
jags.time <- system.time(radon.model <- jags(data=model.names, inits=model.inits, parameters.to.save=parameters.to.save,
                                             model.file=paste(wd,"radon.wishart.bug",sep="/"),DIC=FALSE,
                                             n.chains=3, n.iter=20e3, n.burnin=3e3, n.thin=5))
print(jags.time)

pdf("radon.wishart.bugs.pdf")
plot(radon.model)
dev.off()

radon.summary <- radon.model[["BUGSoutput"]][["summary"]]
radon.means <- radon.summary[,"mean",drop=F]
radon.coefs <- cbind(radon.means[1:J],radon.means[1:J +J])
e.y <- radon.model[["BUGSoutput"]][["sims.matrix"]][,grep('e.y', colnames(radon.model[["BUGSoutput"]][["sims.matrix"]]))]
radon.rsq <- 1 - mean(apply(e.y,1,var)) / var(y)

write.table(radon.coefs,"radon.coefs.from.bugs.csv",sep=",",row.names=F,col.names=F)
