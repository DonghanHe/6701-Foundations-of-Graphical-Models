import numpy as np
import scipy.stats as st

def theta_sample(z,alpha,K):
    n = len(z)
    dparams = np.ones(K)*alpha
    for i in range(n):
        dparams[z[i]] += 1
    theta_samp = st.dirichlet.rvs(dparams)
    return(theta_samp.flatten())


def beta_mean_sample_mn(x, z, beta_sigma2, gauss_mean_params):
    mu_0, sigma2_0 = gauss_mean_params
    mu_0 = np.array(mu_0)
    sigma2_0 = np.array(sigma2_0)
    n, D = np.shape(x)
    K, _ = np.shape(beta_sigma2)

    beta_mean_samp = np.zeros_like(beta_sigma2)
    for k in range(K):
        x_k = x[z == k]
        n_k = len(x_k)
        sigma2_k = beta_sigma2[k, :]

        # post_sigma2_k = np.linalg.inv(np.diag(1/sigma2_0)+np.diag(n_k/sigma2_k))
        post_sigma2_k = (np.diag(1 / (1 / sigma2_0 + n_k / sigma2_k)))
        post_mu_k = np.dot(post_sigma2_k, ((np.eye(D) / sigma2_0) @ mu_0 + (np.eye(D) / sigma2_k) @ sum(x_k)))

        beta_mean_samp[k] = st.multivariate_normal.rvs(mean=post_mu_k, cov=(post_sigma2_k))
    return (beta_mean_samp)


def beta_bernoulli_sample(x, z, beta_hyper_params):
    alpha, beta = beta_hyper_params
    n, D = np.shape(x)
    K = int(max(z) + 1)

    bernoulli_sample = np.zeros((K, D))

    for k in range(K):
        x_k = x[z == k]
        n_k = len(x_k)
        sum_k = np.sum(x_k, axis=0)

        bernoulli_sample[k, :] = st.beta.rvs(alpha + sum_k, beta + n_k - sum_k)

    return (bernoulli_sample)


def beta_mean_sample(x, z, beta_sigma2, gauss_mean_params):
    mu_0, sigma2_0 = gauss_mean_params
    n, D = np.shape(x)
    K, _ = np.shape(beta_sigma2)

    beta_mean_samp = np.zeros_like(beta_sigma2)
    for k in range(K):
        x_k = x[z == k]
        n_k = len(x_k)
        sigma2_k = beta_sigma2[k]

        post_sigma2_k = 1 / (n_k / sigma2_k + 1 / sigma2_0)
        post_mu_k = post_sigma2_k * (mu_0 / sigma2_0 + sum(x_k) / sigma2_k)

        beta_mean_samp[k] = st.norm.rvs(loc=post_mu_k, scale=np.sqrt(post_sigma2_k))
    return (beta_mean_samp)


def z_sample(x, z, theta, beta_mean, beta_sigma2):
    n, D = np.shape(x)
    K = len(theta)

    log_prior = np.log(theta)
    log_likelihood = st.norm.logpdf(x.reshape((n, 1)),
                                    loc=beta_mean.reshape((1, K)),
                                    scale=np.sqrt(beta_sigma2).reshape((1, K)))
    log_prob = log_prior + log_likelihood
    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    prob_norm = prob / np.sum(prob, axis=1).reshape((n, 1))
    # [print(prob_norm[i,:]) for i in range(n)]

    z_samp = np.array([np.random.choice(list(range(K)), replace=True,
                                        p=prob_norm[i, :]) for i in range(n)])
    return (z_samp)


def z_sample_ber(x, z, theta, beta_bernoulli):
    n, D = np.shape(x)
    K = len(theta)

    log_prior = np.log(theta)
    log_likelihood = np.zeros((n, K))
    for k in range(K):
        log_likelihood[:, k] = np.sum(st.bernoulli.logpmf(x, p=beta_bernoulli[k, :]), axis=1)
    log_prob = log_prior + log_likelihood
    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    prob_norm = prob / np.sum(prob, axis=1).reshape((n, 1))
    z_samp = np.array([np.random.choice(list(range(K)), replace=True,
                                        p=prob_norm[i, :]) for i in range(n)])
    return (z_samp)


def z_sample_mn(x, z, theta, beta_mean, beta_sigma2):
    n, D = np.shape(x)
    K = len(theta)

    log_prior = np.log(theta)
    log_likelihood = np.zeros((n, K))
    for k in range(K):
        log_likelihood[:, k] = st.multivariate_normal.logpdf(x,
                                                             mean=beta_mean[k, :], cov=np.diag(beta_sigma2[k, :]))
    log_prob = log_prior + log_likelihood
    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    prob_norm = prob / np.sum(prob, axis=1).reshape((n, 1))
    # [print(prob_norm[i,:]) for i in range(n)]

    z_samp = np.array([np.random.choice(list(range(K)), replace=True,
                                        p=prob_norm[i, :]) for i in range(n)])
    return (z_samp)