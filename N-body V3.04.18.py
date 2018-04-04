# -*- coding: utf-8 -*-
"""
Редактор Spyder

@author: Дмитрий Мелкозеров
"""

# v Подключаемые пакеты v
#===========================================================================
import math as m
import time
import random as r
import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D as p3
#from matplotlib import animation
#===========================================================================
# ^ Подключаемые пакеты ^
# v Константы v
#===========================================================================
# Средняя масса наблюдаемых объектов и их пекулярная скорость
#M_avg = 1.98892 * pow(10, 41) #кг
#V_avg = 4 * pow(10, 5) / np.sqrt(3) #м/с
M_avg = pow(10, 11) #масс Солнц
V_avg = 1.3 * pow(10, -2) / np.sqrt(3) #кпк/(10^12 c)

# Минимальный размер ячейки по одной оси координат
#Distance = 2 * 3.08567758 * pow(10, 22) #м
Distance = 5 * pow(10, 3) #кпк

# Временной интервал
#time_step = pow(10, 13) #с
time_step = 10 #10^12 с

# Гравитационная постоянная
#G = 6.67408313 * pow(10, -11) #м^3/(кг*с^2)
G = 4.51811511 * pow(10, -15) #кпк^3/(М_(Солнца)* (10^12 с)^2)
#===========================================================================
# ^ Константы ^
# v Параметры системы v
#===========================================================================
# Задаем первоначальный размер системы в единицах "Distance"
# для функции parameters_test
i_test = 2
j_test = 2
k_test = 2

# Количество ячеек по одной оси координат (для tree codes)
# ОБЯЗАТЕЛЬНО должно быть в виде 2^(целое положительное число)
n = 16
# Количество частиц
N = 10000
# Число шагов
Steps = 1
#===========================================================================
# ^ Параметры системы ^
# v Используемые функции v
#===========================================================================
# Подфункция, позволяющая сгенерировать определенные
# параметры для тела
def parameters_test(h, p, l):
    x = Distance * h
    y = -Distance * (j_test-1) / 2 + Distance * p
    z = 0 * l
#       Распределение скоростей и масс считаем нормальным
    Vx = 0
    Vy = 0
    Vz = V_avg
    mass = abs(r.normalvariate(M_avg, 0.25*M_avg))
    Sum = np.array([x, y, z, Vx, Vy, Vz, mass])
    return Sum
#_____________________________________________________________________________
# Подфункция, позволяющая сгенерировать случайные параметры для тела (3.04.18)
def randomize_parameters():
    x = r.random()*n*Distance
    y = r.random()*n*Distance
    z = r.random()*n*Distance
#   Распределение скоростей и масс считаем нормальным
#   (пока что квадратичное отклонение выбрано наугад)
    Vx = r.normalvariate(0, 4) * V_avg
    Vy = r.normalvariate(0, 4) * V_avg
    Vz = r.normalvariate(0, 4) * V_avg
    mass = abs(r.normalvariate(M_avg, 0.5*M_avg))
    Sum = np.array([x, y, z, Vx, Vy, Vz, mass, 0])
    return Sum
#_____________________________________________________________________________
# Функция, создающая i*j*k тел
def birth_test():
#   Сначала создаем массив нулей, а затем заполняем его;
#   тела находятся по первому индексу, параметры - по второму
    Test_particles = np.zeros((i_test*j_test*k_test, 7))
    Num = 0
    for l in range(k_test):
        for p in range(j_test):
            for h in range(i_test):
                Test_particles[Num] = parameters_test(h, p, l)
                Num += 1
    return Test_particles
#_____________________________________________________________________________
#   Функция, создающая "body_count" тел
def birth_random(body_count):
#   Сначала создаем массив нулей, а затем заполняем его;
#   тела находятся по первому индексу, параметры - по второму
    Random_particles = np.zeros((body_count, 8))
    for l in range(body_count):
        Random_particles[l] = randomize_parameters()
    return Random_particles
#_____________________________________________________________________________
# Функция, которая выдает растояние между частицам 1 и 2
def Part_distance(Particle_1, Particle_2, Number_1, Nubmer_2):
    delta_x = Particle_1[Number_1, 0] - Particle_2[Nubmer_2, 0]
    delta_y = Particle_1[Number_1, 1] - Particle_2[Nubmer_2, 1]
    delta_z = Particle_1[Number_1, 2] - Particle_2[Nubmer_2, 2]
    return m.sqrt(m.pow(delta_x, 2) + m.pow(delta_y, 2) + m.pow(delta_z, 2))
#_____________________________________________________________________________
# Ускорение по Ньютону
def G_force_Newton(Particles, l, h):
    a = 0
    for p in range(N):
        if not p == l:
            a = a + Particles[p, 6] * (Particles[l, h] - Particles[p, h]) \
                / m.pow(Part_distance(Particles, Particles, l, p), 3)
    a = -G * a
    return a
#_____________________________________________________________________________
# Ньютоновская гравитация, прямой метод
def N_body_direct(X0):
    V = np.zeros((np.size(X0, 0), 7))
    A = np.zeros((np.size(X0, 0), 7))
    a = 0
    for l in range(N):
        for h in range(3):
            a = G_force_Newton(X0, l, h)
            V[l, h] = X0[l, h+3] + a*time_step / 2
            A[(l, h+3)] = a
    X0 += (V+A) * time_step
    return X0
#_____________________________________________________________________________
#   Распределение X_size частиц по ячейкам со стороной Distance
#   с последующей сортировкой по номерам ячеек (3.04.18)
def Distribution(X0, X_size):
    for N_local in range(X_size):
        n_x = int(m.floor(X0[N_local, 0] / Distance))
        n_y = int(m.floor(X0[N_local, 1] / Distance))
        n_z = int(m.floor(X0[N_local, 2] / Distance))
        X0[N_local, 7] = n_x * n * n + n_y * n + n_z
    return X0[X0[:, 7].argsort(kind='mergesort')]
#_____________________________________________________________________________
# Функция, вычисляющая параметры самых малых ячеек из параметров
# находящихся внутри частиц (3.04.18)
def Particles_to_cell(Y, Y_size, order_n):
    n_total = int(m.pow(n, 3))
    R_local = np.zeros([n_total, 7])
    part_num = 0
    part_count = 0
    for cell_num in range(n_total):
        R = np.zeros([6])
        if not part_num == Y_size:
            while Y[part_num, 7] == cell_num:
                R[0:3] += Y[part_num, 0:3] * Y[part_num, 6]
                R[3] += Y[part_num, 6]
                part_num += 1
                if part_num == Y_size:
                    break
        R[4] = part_count
        R[5] = part_num
        part_count = part_num
        if not R[3] == 0:
            R[0:3] = R[0:3] / R[3]
        R_local[cell_num] = [R[0], R[1], R[2], order_n, \
                    R[4], R[5], R[3]]
    return R_local
#_____________________________________________________________________________
# Функция, вычисляющая параметры ячеек за счет
# находящихся внутри ячеек с меньшим порядком (3.04.18)
def Cells_to_cell(R_final, order_n):
    n_linear = int(m.pow(2, order_n))
    n_total = int(m.pow(n_linear, 3))
    R_local = np.zeros([n_total, 7])
    for cell_num in range(n_total):
        R = np.zeros([4])
        cell_x = cell_num // (n_linear * n_linear)
        cell_y = (cell_num // n_linear) - cell_x * n_linear
        cell_z = cell_num % n_linear
        cell_num_0 = int(cell_x * n_linear * n_linear * 8 \
                         + cell_y * n_linear * 4 + cell_z * 2)
        Numbers = [cell_num_0, cell_num_0 + 1, cell_num_0 + n_linear,   \
                   cell_num_0 + n_linear + 1,                           \
                   cell_num_0 + int(m.pow(n_linear, 2)),                \
                   cell_num_0 + int(m.pow(n_linear, 2)) + 1,            \
                   cell_num_0 + int(m.pow(n_linear, 2)) + n_linear,     \
                   cell_num_0 + int(m.pow(n_linear, 2)) + n_linear + 1]
        for u in range(8):
            R[0:3] += R_final[int(Numbers[u]), 0:3] \
                    * R_final[int(Numbers[u]), 6]
            R[3] += R_final[int(Numbers[u]), 6]
        if not R[3] == 0:
            R[0:3] = R[0:3] / R[3]
        R_local[cell_num] = [R[0], R[1], R[2], order_n, \
                    0, 0, R[3]]
    return np.vstack((R_local, R_final))
#_____________________________________________________________________________
# Функция, рассчитывающая ускорение частицы под номером Part_num,
# полученное за счет гравитационного мультипольного взаимодействия с
# частицами в ячейке с номером cell_num.
#(Для использования в методе  Tree code)
def Multipole_Newton(Particles, Mass_center, Part_num, cell_num, Acceleration):
    r_3 = m.pow(Part_distance(Particles, Mass_center, Part_num, cell_num), 3)
    Acceleration[Part_num, 0] += - G * Mass_center[cell_num, 6] \
                * (Particles[Part_num, 0] - Mass_center[cell_num, 0]) / r_3
    Acceleration[Part_num, 1] += - G * Mass_center[cell_num, 6] \
                * (Particles[Part_num, 1] - Mass_center[cell_num, 1]) / r_3
    Acceleration[Part_num, 2] += - G * Mass_center[cell_num, 6] \
                * (Particles[Part_num, 2] - Mass_center[cell_num, 2]) / r_3
    return Acceleration
#_____________________________________________________________________________
# Функция, рассчитывающая ускорение частицы под номером Part_num,
# полученное за счет гравитационного взаимодействия с частицами
# в ячейке с номером cell_num. (Для использования в методе  Tree code)
# (3.04.18)
def Direct_Newton(Particles, Part_num, cell_num, Mass_center, Acceleration):
    a_x = 0
    a_y = 0
    a_z = 0
    for num in range(int(Mass_center[cell_num, 4]), \
                     int(Mass_center[cell_num, 5])):
        if not num == Part_num:
            r_3 = m.pow(Part_distance(Particles, Particles, Part_num, num), 3)
            a_x = a_x + Particles[num, 6] \
                * (Particles[Part_num, 0] - Particles[num, 0]) / r_3
            a_y = a_y + Particles[num, 6] \
                * (Particles[Part_num, 1] - Particles[num, 1]) / r_3
            a_z = a_z + Particles[num, 6] \
                * (Particles[Part_num, 2] - Particles[num, 2]) / r_3
            Acceleration[Part_num, 0] += - G * a_x
            Acceleration[Part_num, 1] += - G * a_y
            Acceleration[Part_num, 2] += - G * a_z
    return Acceleration
#_____________________________________________________________________________
# Функция, определяющая тип взаимодействия (частица-частица,
# мультиполь-частица), а так же раскладывающая большой куб
# на кубы меньшего размера, если удовлетворяется условие
# раскрытия ячейки (3.04.18)
#ПРОВЕРИТЬ НА ОШИБКИ
def Calculate_cell(Particles, Mass_center,      \
                   Part_num, cell_num,          \
                   order_count, Acceleration):
    n_linear = int(m.pow(2, Mass_center[cell_num, 3]))
    L = n * Distance / n_linear
    if not Mass_center[cell_num, 6] == 0:
        if Part_distance(Particles, Mass_center,                    \
                         Part_num, cell_num) > (L * m.sqrt(3)):
            Acceleration = Multipole_Newton(Particles, Mass_center, \
                                                Part_num, cell_num, \
                                                Acceleration)
        else:
            if n_linear == n:
                Acceleration = Direct_Newton(Particles, Part_num,   \
                                             cell_num, Mass_center,  \
                                             Acceleration)
            else:
                order_count += m.pow(n_linear, 3)
                Calculate_cube(Particles, Mass_center,  \
                               Part_num, cell_num,      \
                               order_count, Acceleration)
    return Acceleration
#_____________________________________________________________________________
# Функция, представляющая один большой куб как 8 кубов меньшего размера
#ПРОВЕРИТЬ НА ОШИБКИ (3.04.18)
def Calculate_cube(Particles, Mass_center,  \
                     Part_num, cell_num,    \
                     order_count, Acceleration):
    n_linear = int(m.pow(2, Mass_center[cell_num, 3]))
    cell_x = cell_num // (n_linear * n_linear)
    cell_y = (cell_num // n_linear) - cell_x * n_linear
    cell_z = cell_num % n_linear
    cell_num_0 = int(order_count + (cell_x * n_linear * n_linear \
                         + cell_y * n_linear + cell_z) * 2)
    cell_num_1 = cell_num_0 + 1
    cell_num_2 = cell_num_0 + n_linear
    cell_num_3 = cell_num_2 + 1
    cell_num_4 = cell_num_0 + int(m.pow(n_linear, 2))
    cell_num_5 = cell_num_4 + 1
    cell_num_6 = cell_num_4 + n_linear
    cell_num_7 = cell_num_6 + 1
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_0,     \
                                  order_count, Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_1,     \
                                  order_count, Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_2,     \
                                  order_count, Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_3,     \
                                  order_count, Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_4,     \
                                  order_count, Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_5,     \
                                  order_count, Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_6,     \
                                  order_count, Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_7,     \
                                  order_count, Acceleration)
    return Acceleration
#_____________________________________________________________________________
# Функция, подщитывающая ускорение для каждой частицы (3.04.18)
def Interaction(Particles, Mass_center):
    Number_of_particles = np.size(Particles, 0)
    Acceleration = np.zeros([Number_of_particles, 3])
    for Part_num in range(Number_of_particles):
        Acceleration = Calculate_cube(Particles, Mass_center, \
                     Part_num, 0, 0, Acceleration)
    return Acceleration
#_____________________________________________________________________________
# Функция, позволяющая получить новые параметры частиц
# из матрицы Y с помощью метода Tree code
def Tree_code_gravity(Y):
    order_n = int(m.log2(n))
    Y_size = np.size(Y, 0)
#    start = time.time()
    Y = Distribution(Y, Y_size)
#    computing_time = time.time() - start
#    print("Сортировка", computing_time, "с")
#    start = time.time()
    R_final = Particles_to_cell(Y, Y_size, order_n)
    while order_n > 1:
        order_n += -1
        R_final = Cells_to_cell(R_final, order_n)
#    computing_time = time.time() - start
#    print("Работа с ячейками", computing_time, "с")
#    start = time.time()
    A = Interaction(Y, R_final)
#    computing_time = time.time() - start
#    print("Рассчет взаимодействия", computing_time, "с")
    Y[:, 0:3] += Y[:, 3:6] * time_step + A[:, 0:4] * m.pow(time_step, 2) / 2
    Y[:, 3:6] += A[:, 0:4] * time_step
    return A
#_____________________________________________________________________________
# Функция для "скирншота" положения всех частиц
def screenshot(System_parameters, Name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = System_parameters[:, 0]
    y = System_parameters[:, 1]
    z = System_parameters[:, 2]
    ax.scatter(x, y, z, color='red', s = 1)
    ax.autoscale(False)
#    Для осей в единицах СИ
#    ax.set_xlabel('x, м')
#    ax.set_ylabel('y, м')
#    ax.set_zlabel('z, м')
    ax.set_xlabel('x, кпк')
    ax.set_ylabel('y, кпк')
    ax.set_zlabel('z, кпк')
    plt.savefig(Name, dpi=1280)
    plt.show()
#===========================================================================
# ^ Используемые функции ^
# v Область с исполняемым кодом v
#===========================================================================
#X = birth_()
X = birth_random(N)

start = time.time()
for q in range(Steps):
    X = Tree_code_gravity(X)
computing_time = time.time() - start
print("Полное время вычислений", computing_time, "с")
#print(X)

#Name = "Начальное положение.png"
#screenshot(X, Name)
#
#start = time.time()
#
#for q in range(Steps):
#    X = N_body_direct(X)
##    if l in [50, 100, 150, 200, 250, 300, 350, 400, 450]:
##        screenshot(X, str(l))
##    print(l)
#
#computing_time = time.time() - start
#print("Полное время вычислений", computing_time, "с")

#Name = "Итоговое положение.png"
#screenshot(X, Name)
#===========================================================================
# ^ Область с исполняемым кодом ^
