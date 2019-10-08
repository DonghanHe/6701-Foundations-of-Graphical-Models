import numpy as np
import scipy.stats as st

def theta_init(K):
    return(np.ones(K)/K)
def beta_ber_init(x,K):
    n,_ = x.shape
    return(np.ones((K,n))*0.5)
def beta_mean_sampleinit(x,K):
    n,_ = np.shape(x)
    i_sample = np.random.choice(list(range(n)),size=K, replace=False)
    x_sample = x[i_sample,:]
    return(x_sample)
def beta_sigma2_init(x,K):
    x_var = np.var(x)
    return(np.ones(K)*x_var/(K**2)*(st.uniform.rvs(0.5,1,size=K)))
def z_init(n,K):
    pre_z = st.multinomial.rvs(n=1,p=np.ones(K)/K,size=n)
    z_sample = np.array([np.where(pre_z[i,:]) for i in range(n)]).flatten()
    return(z_sample)