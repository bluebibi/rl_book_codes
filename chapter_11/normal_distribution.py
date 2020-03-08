import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

legend = []


def norm(mu, var):
    legend.append("N(mean={0}, std.={1})".format(mu, var))
    return stats.norm(mu, var).pdf(x)


x = np.linspace(-10, 10, 101)
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(16, 12))
mean_and_std = [(0, 1), (0, 0.7), (0, 0.5), (2, 1), (-2, 1)]
for i in np.arange(5):
    plt.plot(x, norm(mean_and_std[i][0], mean_and_std[i][1]))
plt.xlabel('x')
plt.legend(legend, prop={'size': 20})
plt.grid()
plt.savefig("./normal_distribution.png")
plt.show()