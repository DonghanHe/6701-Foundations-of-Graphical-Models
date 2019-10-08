import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from gibbs_sampling.sampling_functions import *
from gibbs_sampling.initialization import *


def log_joint(data, theta_s, z_s, beta_mean_s, beta_sigma_s, theta_p, beta_mean_p, beta_sigma_p):
    try:
        n, D = np.shape(data)
    except ValueError:
        n = len(data)
        D = 1
    x = data
    K = len(theta_s)

    # print(theta_s.shape,theta_p.shape)
    log_prior_dir = st.dirichlet.logpdf(theta_s, theta_p)
    z_s_1vec = np.zeros((n, K))
    z_s = np.array(z_s, dtype=int)
    z_s_1vec[np.arange(n), z_s] = 1
    log_prior_z = st.multinomial.logpmf(z_s_1vec, 1, theta_s)
    log_prior_gau = np.sum(st.multivariate_normal.logpdf(beta_mean_s,
                                                         mean=beta_mean_p[0], cov=beta_mean_p[1]))
    result = log_prior_dir + log_prior_gau

    # for i in range(n):
    #    result += np.log(theta_s[z_s[i]])+st.multivariate_normal.logpdf(x[i,:],mean=beta_mean_s[z_s[i],:])

    for k in range(K):
        x_k = x[z_s == k, :]
        result += np.sum(theta_s[k] + st.multivariate_normal.logpdf(x[z_s == k, :], mean=beta_mean_s[k, :]))

    return (result)


def log_joint_ber(data, theta_s, z_s, beta_ber_s, theta_p, beta_ber_p):
    try:
        n, D = np.shape(data)
    except ValueError:
        n = len(data)
        D = 1
    x = data
    K = len(theta_s)

    # print(theta_s.shape,theta_p.shape)
    log_prior_dir = st.dirichlet.logpdf(theta_s, theta_p)
    z_s_1vec = np.zeros((n, K))
    z_s = np.array(z_s, dtype=int)
    z_s_1vec[np.arange(n), z_s] = 1
    log_prior_z = st.multinomial.logpmf(z_s_1vec, 1, theta_s)
    log_prior_ber = np.sum(st.beta.logpdf(beta_ber_s, beta_ber_p[0], beta_ber_p[1]))
    # print(log_prior_ber)
    # print("=====================")
    # print("beta_bernoulli_sample")
    # print(str(beta_ber_s))
    # print("beta_bernoulli_parameters")
    # print(str(beta_ber_p))
    # log_prior_gau = np.sum(st.multivariate_normal.logpdf(beta_mean_s,
    # mean=beta_mean_p[0],cov=beta_mean_p[1]))
    result = log_prior_dir + log_prior_ber

    # for i in range(n):
    #    result += np.log(theta_s[z_s[i]])+st.multivariate_normal.logpdf(x[i,:],mean=beta_mean_s[z_s[i],:])

    for k in range(K):
        x_k = x[z_s == k, :]
        result += np.sum(theta_s[k] + np.sum(st.bernoulli.logpmf(x[z_s == k, :], p=beta_ber_s[k, :]), axis=1))

    return (result)


def gibbs_sampler_ber(data, cat_num ,hyper_theta ,hyper_mean ,T=5e2 ,plot=False,
                      full = False):
    try:
        n ,D = np.shape(data)
    except ValueError:
        n = len(data)
        D = 1
    x = data
    K = cat_num
    T = int(T)

    if full:
        theta_stored = np.zeros((K ,T))
        beta_ber_stored = np.zeros((K ,D ,T))
        z_stored = np.zeros((n ,T))
    theta = theta_init(K)
    beta_ber = beta_ber_init(x ,K)
    z = z_init(n ,K)
    if plot:
        log_p = np.zeros(T)

    for t in range(T):
        theta = theta_sample(z ,hyper_theta ,K)
        beta_ber = beta_bernoulli_sample(x ,z ,hyper_mean)
        z = z_sample_ber(x ,z ,theta ,beta_ber)

        if full:
            theta_stored[: ,t] = theta
            beta_ber_stored[: ,: ,t] = beta_ber
            z_stored[: ,t] = z
        if plot:
            log_p[t] = log_joint_ber(data ,theta ,z ,beta_ber ,hyper_theta ,hyper_mean)
            # log_joint(data,theta,z,beta_mean,beta_sigma,theta_p,beta_mean_p,beta_sigma_p)

    if plot:
        print(log_p)
        plt.plot(log_p)
        plt.title("log joint probability")
        plt.xlabel('iterations')
        plt.ylabel('log joint probability')
        # plt.yscale('symlog')
        plt.show()
    if full:
        return({'z' :z_stored, 'theta' :theta_stored, 'beta_ber' :beta_ber_stored})
    else:
        return({'z' :z ,'theta' :theta ,'beta_ber' :beta_ber})


def gibbs_sampler_mn(data, cat_num, hyper_theta, hyper_mean, hyper_sigma2, T=5e2, plot=False,
                     full=False):
    try:
        n, D = np.shape(data)
    except ValueError:
        n = len(data)
        D = 1
    x = data
    K = cat_num
    T = int(T)

    theta_stored = np.zeros((K, T))
    beta_mean_stored = np.zeros((K, D, T))
    beta_sigma2_stored = np.zeros((K, D, T))
    z_stored = np.zeros((n, T))
    theta = theta_init(K)
    beta_mean = beta_mean_sampleinit(x, K)
    beta_sigma2 = np.ones((K, D))
    z = z_init(n, K)
    if plot:
        log_p = np.zeros(T)

    for t in range(T):
        theta = theta_sample(z, hyper_theta, K)

        beta_mean = beta_mean_sample_mn(x, z, beta_sigma2, hyper_mean)

        z = z_sample_mn(x, z, theta, beta_mean, beta_sigma2)

        if full:
            theta_stored[:, t] = theta
            beta_mean_stored[:, :, t] = beta_mean
            z_stored[:, t] = z
        if plot:
            log_p[t] = log_joint(data, theta, z, beta_mean, beta_sigma2, hyper_theta, hyper_mean, hyper_sigma2)
            # log_joint(data,theta,z,beta_mean,beta_sigma,theta_p,beta_mean_p,beta_sigma_p)

    if plot:
        print(log_p)
        plt.plot(log_p)
        plt.title("log joint probability")
        plt.xlabel('iterations')
        plt.ylabel('log joint probability')
        # plt.yscale('symlog')
        plt.show()
    if full:
        return ({'z': z_stored, 'theta': theta_stored, 'beta_mean': beta_mean_stored})
    else:
        return ({'z': z, 'theta': theta, 'beta_mean': beta_mean})