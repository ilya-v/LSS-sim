
QX = 0
QY = 1
QZ = 2

QVX = 3
QVY = 4
QVZ = 5

QMASS = 6

QAX = 7
QAY = 8
QAZ = 9

QPHI = 10
QCELLIDX = 11
QT = 12
QU = 13

QCOORD = slice(QX, QZ + 1)
QVEL = slice(QVX, QVZ + 1)
QACC = slice(QAX, QAZ + 1)


def velocity(X):
    return X[:, QVEL]

def position(X):
    return X[:, QCOORD]
