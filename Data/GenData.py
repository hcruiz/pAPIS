import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt

dim = 1
dt = 0.01
T = 3.0
variance = 1
var_obs = 1
steps = int(T/dt)

dim_obs = dim
obs_step = 10
obs_time = dt*np.arange(obs_step,steps+1,obs_step)
observations = np.zeros([len(obs_time),dim_obs])
x = np.zeros([steps+1,dim])

for t in range(steps):
	x[t+1,:] = x[t,:] + np.sqrt(dt*variance)*rnd.randn()
	if np.any(dt*(t+1) == obs_time):
		observations[dt*(t+1) == obs_time,:] = x[t+1,:] + np.sqrt(var_obs)*rnd.randn()
	
#plt.plot(dt*np.arange(steps),x)
#plt.plot(obs_time,observations,'r*')
#plt.show()
np.savez('time_series.npz',obs_time = obs_time, observations = observations)
print 'Data generated: '
print 'Data: ', observations.T
print 'Time points: ', obs_time
