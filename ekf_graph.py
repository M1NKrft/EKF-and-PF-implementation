import numpy as np
import matplotlib.pyplot as plt
from soccer_field import Field
import policies
from ekf import ExtendedKalmanFilter
import localization
initial_cov = np.diag([10, 10, 1])
alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
beta = np.diag([np.deg2rad(5)**2])
for r in [-6,-4,-2,2,4,6]:
    xpoints=np.array([])
    ypoints=np.array([])
    for i in range(10):
        filt=ExtendedKalmanFilter(
            np.array([180, 50, 0]).reshape((-1, 1)),
            np.diag([10, 10, 1]),
            (2**r) * alphas,
            (2**r) *beta)
        env = Field((2**r) *alphas, (2**r) *beta)
        policy = policies.OpenLoopRectanglePolicy()
        mean=localization.localize(env,policy,filt,np.array([180, 50, 0]).reshape((-1, 1)),200)[0]
        mean_pos=mean.mean()
        xpoints=np.append(xpoints,[r])
        ypoints=np.append(ypoints,[mean_pos])
    plt.plot(xpoints,ypoints,'o')
plt.xlabel("log2(r)")
plt.ylabel("Mean Position Error")
plt.show(block=True)
plt.savefig('ekfplotb.png')
