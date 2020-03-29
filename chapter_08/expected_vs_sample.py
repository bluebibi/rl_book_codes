import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

def b_to_error(b):
    errors = []

    for t in range(1, 2 * b + 1):
        errors.append(np.sqrt((b - 1) / (b * t)))

    return errors


def expected_vs_sample():
    runs = 10
    branch = [2, 10, 100, 1000, 10000]
    for b in branch:
        rms_errors = np.zeros((runs, 2 * b))
        for r in tqdm(np.arange(runs)):
            rms_errors[r] = b_to_error(b)

        rms_errors = rms_errors.mean(axis=0)
        x_vals = (np.arange(len(rms_errors)) + 1) / float(b)
        plt.plot(x_vals, rms_errors, label='b = {0}'.format(b), linestyle="--")

    x_vals = [0., 1.0, 2.0]
    plt.plot(x_vals, [1.0, 0.0, 0.0], label='expected update')

    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("$\max_{a'}Q(s', a')$ 계산 횟수")
    plt.xticks(x_vals, ['0', 'b', '2b'])
    plt.ylabel('RMS 오차')
    plt.legend()

    plt.savefig('images/expected_vs_sample.png')
    plt.close()


if __name__ == '__main__':
    expected_vs_sample()