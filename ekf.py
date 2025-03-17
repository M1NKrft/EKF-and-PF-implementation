import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.
        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        mu_bar=env.forward(self.mu,u)
        Gt=env.G(self.mu,u)
        Vt = env.V(self.mu, u)
        noise = env.noise_from_motion(u, self.alphas)
        Rt = np.dot(np.dot(Vt, noise), np.transpose(Vt))
        sigma_bar=np.dot(np.dot(Gt,self.sigma),np.transpose(Gt)) + Rt
        Ht=env.H(mu_bar, marker_id) 
        Ht_T= (np.transpose(Ht)).reshape(-1,1)
        invmat= np.linalg.inv((np.dot(np.dot(Ht,sigma_bar),Ht_T) + self.beta))
        Kt= np.dot(np.dot(sigma_bar,Ht_T),invmat)
        obs_diff= minimized_angle(z - env.observe(mu_bar,marker_id))
        self.mu = np.ravel(mu_bar + np.dot(Kt,obs_diff)).reshape(-1,1)
        self.sigma= np.dot(np.identity(3) - np.dot(Kt,Ht),sigma_bar)
        self.mu[2]= minimized_angle(self.mu[2])
        return self.mu, self.sigma