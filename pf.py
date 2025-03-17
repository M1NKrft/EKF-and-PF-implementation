import numpy as np

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.
        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        # Predict step
        prediction = np.zeros((self.num_particles, 3))
        total_weight = 0
        weights = np.ones(self.num_particles)
        for i in range(self.num_particles):
            sample_noisyaction = env.sample_noisy_action(u, self.alphas)
            sample = env.forward(self.particles[i, :], sample_noisyaction)
            obs_model = env.observe(sample, marker_id)
            innovation = minimized_angle(z - obs_model)
            weights[i] = env.likelihood(innovation, self.beta)
            prediction[i, :] = sample.reshape(1, -1)
            total_weight +=weights[i]
        weights /= total_weight
        self.particles, self.weights = self.resample(prediction, weights)
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.
        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        new_particles, new_weights = particles, weights
        # YOUR IMPLEMENTATION HERE
        n = len(particles)
        indices = np.zeros(n, dtype=int)
        r = np.random.uniform(0, 1/n)
        c = weights[0]
        i = 0

        for m in range(0, n):
            U = r + m /n
            while U > c:
                i += 1
                c += weights[i]
            indices[m] = i

        new_particles = particles[indices]
        new_weights = np.ones(n) / n
        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov