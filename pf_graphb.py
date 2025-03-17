import numpy as np
import matplotlib.pyplot as plt
from soccer_field import Field
import policies
from utils import minimized_angle, plot_field, plot_robot, plot_path
from pf import ParticleFilter
def localize(env, policy, filt, x0, num_steps, plot=False):
    # Collect data from an entire rollout
    states_noisefree, states_real, action_noisefree, obs_noisefree, obs_real = \
            env.rollout(x0, policy, num_steps)
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)

    if plot:
        fig = env.get_figure()

    for i in range(num_steps):
        x_real = states_real[i+1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            # filters only know the action and observation
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        states_filter[i+1, :] = mean.ravel()

        if plot:
            fig.clear()
            plot_field(env, marker_id)
            plot_robot(env, x_real, z_real)
            plot_path(env, states_noisefree[:i+1, :], 'g', 0.5)
            plot_path(env, states_real[:i+1, :], 'b')
            if filt is not None:
                plot_path(env, states_filter[:i+1, :2], 'r')
            fig.canvas.flush_events()

        errors[i, :] = (mean - x_real).ravel()
        errors[i, 2] = minimized_angle(errors[i, 2])
        position_errors[i] = np.linalg.norm(errors[i, :2])

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            print('Badly conditioned cov (setting to identity):', cond_number)
            print(cov)
            cov = np.eye(3)
        mahalanobis_errors[i] = \
                errors[i:i+1, :].dot(np.linalg.inv(cov)).dot(errors[i:i+1, :].T)

    mean_position_error = position_errors.mean()
    mean_mahalanobis_error = mahalanobis_errors.mean()
    anees = mean_mahalanobis_error / 3

    if filt is not None:
        print('-' * 80)
        print('Mean position error:', mean_position_error)
        print('Mean Mahalanobis error:', mean_mahalanobis_error)
        print('ANEES:', anees)

    if plot:
        plt.show(block=True)

    return mean_position_error
alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
beta = np.diag([np.deg2rad(5)**2])
for r in [-6,-4,-2,2,4,6]:
    xpoints=np.array([])
    ypoints=np.array([])
    for i in range(10):
        env = Field((2**r) *alphas, (2**r) *beta)
        policy = policies.OpenLoopRectanglePolicy()
        filt=ParticleFilter(
            np.array([180, 50, 0]).reshape((-1, 1)),
            np.diag([10, 10, 1]),
            100,
            (2**r) * alphas,
            (2**r) *beta)
        mean_pos=localize(env, policy, filt, np.array([180, 50, 0]).reshape((-1, 1)), 200)
        xpoints=np.append(xpoints,[r])
        ypoints=np.append(ypoints,[mean_pos])
    plt.plot(xpoints,ypoints,'o')
plt.xlabel("log2(r)")
plt.ylabel("Mean Position Error")
plt.show(block=True)
plt.savefig('plot.png')
