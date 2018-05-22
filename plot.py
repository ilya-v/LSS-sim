import matplotlib.pyplot as plt
import telemetry as tm
from layout import *

LABELS = {
    tm.Report.Q_POT_ENERGY: 'Potential energy',
    tm.Report.Q_KIN_ENERGY: 'Kinetic energy',
    tm.Report.Q_MAX_DELTA_KIN: 'Max kinetic energy difference per step',
    tm.Report.Q_MAX_DELTA_POT: 'Max potential energy difference per step',
    tm.Report.Q_DELTA_TOT: 'Total energy difference per step'
}


def file_name(title):
    return title.replace(' ', '_')


def draw(title, report, *params_q):
    fig = plt.figure()
    id_subplot = 1
    subplot = None
    for param_q in params_q:
        subplot = fig.add_subplot(id_subplot)
        subplot.plot(report.data[0:report.n_lines, 0],
                     report.data[0:, param_q])
        subplot.set_ylabel(LABELS[param_q])
        id_subplot += 1
    if subplot:
        subplot.set_xlabel('Step')
        subplot.set_title(title)
    plt.savefig(file_name(title), dpi=640)
    plt.show()


def screenshot(X, name, point_size=0.2):
    # Функция для "скирншота" положения всех частиц
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, QX], X[:, QY], X[:, QZ], color='red', s=point_size)
    ax.autoscale(False)
    ax.set_xlabel('x, кпк')
    ax.set_ylabel('y, кпк')
    ax.set_zlabel('z, кпк')
    plt.savefig(name, dpi=1280)
#    plt.show()
