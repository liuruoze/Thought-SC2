## Import the packages
import numpy as np
from scipy import stats


## Define 2 random distributions
#Sample Size
N = 10
#Gaussian distributed data with mean = 2 and var = 1
a = np.random.randn(N) + 2
#Gaussian distributed data with with mean = 0 and var = 1
b = np.random.randn(N)


## Calculate the Standard Deviation
#Calculate the variance to get the standard deviation

#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

#std deviation
s = np.sqrt((var_a + var_b)/2)
s



## Calculate the t-statistics
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))



## Compare with the critical t-value
#Degrees of freedom
df = 2*N - 2

#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=df)


print("t = " + str(t))
print("p = " + str(2*p))
### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(p2))


print("ACRL test!")
a =  np.array([0.92,0.93,0.93,0.96,0.90])
b =  np.array([0.87,0.00,0.76,0.91,0.87])
c =  np.array([0.87,0.76,0.91,0.87])
b[1] = c.mean()
t2, p2 = stats.ttest_ind(a,b)
print("improve: " + str(a.mean()-b.mean()))
print("t = " + str(t2))
print("p = " + str(p2))

print("Time test!")
a =  np.array([102, 105, 98])
b =  np.array([1.08, 1.25, 1.15, 1.24, 1.15])
t2, p2 = stats.ttest_ind(a,b)
print("improve: " + str(a.mean()/b.mean()))
print("t = " + str(t2))
print("p = " + str(p2))