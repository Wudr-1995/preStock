import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

matplotlib.use('TkAgg')

data = np.loadtxt('./testResult', delimiter=' ', dtype=float)

font = {'weight': 'normal', 'size': 15}

x = range(np.size(data, axis=0))
fig0 = plt.figure()
ax = fig0.add_subplot(1, 1, 1)
ax.set_ylabel('Bias / YUAN', font)
# plt.plot(x, data[:, 0])
# plt.plot(x, data[:, 1])
d = data[:, 0] - data[:, 1]

pre = data[:, 0]
label = data[:, 1]
pre = np.expand_dims(pre, 1)
label = np.expand_dims(label, 0)

m = np.mean(d)
m = m * np.ones(np.size(d))
plt.tick_params(labelsize=15)
plt.plot(x, d - m)

fig1 = plt.figure()
ax = fig1.add_subplot(1, 1, 1)
ax.set_xlabel('Label / YUAN', font)
ax.set_ylabel('Prediction / YUAN', font)
ax.scatter(label, pre, c='r', marker='.', linewidth=3)
plt.tick_params(labelsize=15)
plt.xlim(0, 10)
plt.ylim(0, 10)

plt.show()
