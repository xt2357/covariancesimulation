require(mvtnorm)

N = 10^4 # sample size
rho_vec <- c(0.3,0.5,0.6,0.8)
K_vec <- c(50,100,200,500,1000) # number of buckets
M = 1000 # rep times
result_vec = result_vec_2 = c()
for (rho in rho_vec) {
  cov <- rho*sqrt(25*40)
  a <- matrix(c(25,cov,cov,40),ncol = 2)
  sigma <- t(a) %*% a
  beta_optimal = sigma[1,2]/sigma[1,1]
  newvar_optimal = sigma[2,2]/N + beta_optimal^2*(sigma[1,1]/N) -2*beta_optimal*(sigma[1,2]/N)
  for (K in K_vec) {
    mean_vec = c()
    sd_vec = c()
    for (j in 1:M) {
      # col 1 for Y(A/A), col 2 for X(A/B)
      x <- data.frame(rmvnorm(n= N, mean=c(100,250), sigma=sigma)) 
      x[,3]<- as.vector(sample(1:K,N,replace = TRUE)) 
      aggregate_x <- aggregate(x[,1:2], list(x[,3]), mean)
      n = N/K
      #K_s1_s1_1=sum((aggregate_x[,1] - mean(aggregate_x[,1]))*((aggregate_x[,2] - mean(aggregate_x[,2]))))/(K-1)
      #cov_b = K_s1_s1_1/(n^2)*K
      cov_b = cov(aggregate_x[,2],aggregate_x[,3])/K
      cov_no_bucket = cov(x[,1],x[,2])/N
      beta_b  = cov_b/(var(aggregate_x[,2])/K)
      beta_no_bucket = cov_no_bucket/(var(x[,1])/N)
      newvar_b =  var(aggregate_x[,3])/K + beta_b^2*var(aggregate_x[,2])/K - 2*beta_b*cov_b
      newvar_no_bucket = var(x[,2])/N + beta_no_bucket^2*var(x[,1])/N - 2*beta_no_bucket*cov_no_bucket
      #print(newvar_no_bucket)
      mean_vec = c(mean_vec,abs(beta_b/beta_optimal-1.0))
      }
    result_vec = c(result_vec,mean(mean_vec))
    }
  }

relative_err = matrix(result_vec,ncol=length(K_vec),byrow = TRUE)
colnames(relative_err) <- c(K_vec)
rownames(relative_err) <- rho_vec
relative_err <- as.table(relative_err)

print(relative_err)