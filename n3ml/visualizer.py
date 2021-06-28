import matplotlib.pyplot as plt


class PlotW:
    def __init__(self):
        self.mat = plt.matshow()

    def __call__(self, w):
        pass


def plot_w(fig, mat, w):
    if mat is None or fig is None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mat = ax.matshow(w[0].reshape(28, 28))
        mat.set_clim(0, 1)
        fig.colorbar(mat)
        return fig, mat
    plt.gcf()
    print(w[0].reshape(28, 28))
    mat.set_data(w[0].reshape(28, 28))
    fig.canvas.draw()
    return fig, mat


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 10*np.pi, 100)
    y = np.sin(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'b-')

    for phase in np.linspace(0, 10*np.pi, 100):
        line1.set_ydata(np.sin(0.5 * x + phase))
        fig.canvas.draw()
