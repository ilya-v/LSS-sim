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
#import statistics as stat
#import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as p3
#from matplotlib import animation
#===========================================================================
# ^ Подключаемые пакеты ^
# v Константы v
#===========================================================================
# Средняя масса наблюдаемых объектов и их пекулярная скорость
#M_avg = 1.98892 * pow(10, 41) #кг
#V_avg = 0 #4 * pow(10, 5) / np.sqrt(3) #м/с
M_avg = pow(10, 11) #масс Солнц
V_avg = 0 #1.3 * pow(10, -2) / np.sqrt(3) #кпк/(10^12 c)

# Минимальный размер ячейки по одной оси координат
#Distance = 2 * 3.08567758 * pow(10, 22) #м
Distance = 5 * m.pow(10, 3) #кпк

# Временной интервал
#time_step = pow(10, 13) #с
time_step = 100 #10^12 с

# Гравитационная постоянная
#G = 6.67408313 * m.pow(10, -11) #м^3/(кг*с^2)
G = 4.51811511 * m.pow(10, -15) #кпк^3/(М_(Солнца)* (10^12 с)^2)
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
Steps = 500
#===========================================================================
# ^ Параметры системы ^
# v Используемые функции v
#===========================================================================
# Подфункция, позволяющая сгенерировать определенные
# параметры для тела
def parameters_test(h, p, l):
    x = Distance * (1 / 2 + h / 1)
    y = Distance * (1 / 2 + p / 1)
    z = Distance * (1 / 2 + l / 1)
#       Распределение скоростей и масс считаем нормальным
    Vx = 0
    Vy = 0
    Vz = 0
    mass = abs(M_avg)
    Sum = np.array([x, y, z, Vx, Vy, Vz, mass, 0, 0, 0, 0])
    return Sum
#_____________________________________________________________________________
# Подфункция, позволяющая сгенерировать случайные параметры для тела (3.04.18)
def randomize_parameters():
    x = r.random() * n * Distance
    y = r.random() * n * Distance
    z = r.random() * n * Distance
#   Распределение скоростей и масс считаем нормальным
#   (пока что квадратичное отклонение выбрано наугад)
    Vx = r.normalvariate(0, 4) * V_avg
    Vy = r.normalvariate(0, 4) * V_avg
    Vz = r.normalvariate(0, 4) * V_avg
    mass = abs(r.normalvariate(M_avg, 0.5*M_avg))
    Sum = np.array([x, y, z, Vx, Vy, Vz, mass, 0, 0, 0, 0])
    return Sum
#_____________________________________________________________________________
# Функция, создающая i*j*k тел
def birth_test():
#   Сначала создаем массив нулей, а затем заполняем его;
#   тела находятся по первому индексу, параметры - по второму
    test_particles = np.zeros((i_test * j_test * k_test, 11))
    Num = 0
    for l in range(k_test):
        for p in range(j_test):
            for h in range(i_test):
                test_particles[Num] = parameters_test(h, p, l)
                Num += 1
    return test_particles
#_____________________________________________________________________________
#   Функция, создающая "body_count" тел
def birth_random(body_count):
#   Сначала создаем массив нулей, а затем заполняем его;
#   тела находятся по первому индексу, параметры - по второму
    random_particles = np.zeros((body_count, 11))
    for l in range(body_count):
        random_particles[l] = randomize_parameters()
    return random_particles
#_____________________________________________________________________________
# Функция, которая выдает растояние между частицам 1 и 2
def part_distance(Particle_1, Particle_2, Number_1, Nubmer_2):
    delta_x = Particle_1[Number_1, 0] - Particle_2[Nubmer_2, 0]
    delta_y = Particle_1[Number_1, 1] - Particle_2[Nubmer_2, 1]
    delta_z = Particle_1[Number_1, 2] - Particle_2[Nubmer_2, 2]
    return m.sqrt(delta_x * delta_x     \
                  + delta_y * delta_y   \
                  + delta_z * delta_z)
#_____________________________________________________________________________
# Ускорение по Ньютону
def g_force_Newton(Particles, l, h):
    a = 0
    for p in range(N):
        if not p == l:
            a = a + Particles[p, 6] * (Particles[l, h] - Particles[p, h]) \
                / m.pow(part_distance(Particles, Particles, l, p), 3)
    a = -G * a
    return a
#_____________________________________________________________________________
# Ньютоновская гравитация, прямой метод
def N_body_direct(X0):
    X0[:, 3:6] += X0[:, 7:10] * time_step / 2
    X0[:, 0:3] += X0[:, 3:6] * time_step
    A = np.zeros((np.size(X0, 0), 3))
    a = 0
    for l in range(N):
        for h in range(3):
            a = g_force_Newton(X0, l, h)
            A[(l, h)] = a
    X0[:, 7:10] = A
    X0[:, 3:6] += X0[:, 7:10] * time_step / 2
    return X0
#_____________________________________________________________________________
#   Распределение X_size частиц по ячейкам со стороной Distance
#   с последующей сортировкой по номерам ячеек (3.04.18)
def distribution(X0, X_size):
    for N_local in range(X_size):
        n_x = int(m.floor(X0[N_local, 0] / Distance))
        n_y = int(m.floor(X0[N_local, 1] / Distance))
        n_z = int(m.floor(X0[N_local, 2] / Distance))
        X0[N_local, 10] = n_x * n * n + n_y * n + n_z
    return X0[X0[:, 10].argsort(kind='mergesort')]
#_____________________________________________________________________________
# Функция, вычисляющая параметры самых малых ячеек из параметров
# находящихся внутри частиц (3.04.18)
def particles_to_cell(Y, Y_size, order_n):
    n_total = int(m.pow(n, 3))
    R_local = np.zeros([n_total, 15])
    part_num = 0
    part_count = 0
    for cell_num in range(n_total):
        R = np.zeros([6])
        if not part_num == Y_size:
            while Y[part_num, 10] == cell_num:
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
        R_local[cell_num] = [R[0], R[1], R[2], order_n, R[4], R[5], R[3],   \
                0, 0, 0, 0, 0, 0, 0, 0]
    return R_local
#_____________________________________________________________________________
# Функция, вычисляющая параметры ячеек за счет
# находящихся внутри ячеек с меньшим порядком (5.04.18)
def cells_to_cell(R_final, order_n):
    n_linear = int(m.pow(2, order_n))
    n_total = int(m.pow(n_linear, 3))
    R_local = np.zeros([n_total, 15])
    for cell_num in range(n_total):
        R = np.zeros([4])
        cell_x = cell_num // (n_linear * n_linear)
        cell_y = (cell_num // n_linear) - cell_x * n_linear
        cell_z = cell_num % n_linear
        cell_num_0 = int(cell_x * n_linear * n_linear * 8 \
                         + cell_y * n_linear * 4 + cell_z * 2)
        Numbers = [cell_num_0, cell_num_0 + 1,                              \
                   cell_num_0 + n_linear * 2,                               \
                   cell_num_0 + n_linear * 2 + 1,                           \
                   cell_num_0 + int(m.pow(n_linear * 2, 2)),                \
                   cell_num_0 + int(m.pow(n_linear * 2, 2)) + 1,            \
                   cell_num_0 + int(m.pow(n_linear * 2, 2)) + n_linear * 2, \
                   cell_num_0 + int(m.pow(n_linear * 2, 2)) + n_linear * 2 + 1]
        for u in range(8):
            R[0:3] += R_final[int(Numbers[u]), 0:3] \
                    * R_final[int(Numbers[u]), 6]
            R[3] += R_final[int(Numbers[u]), 6]
        if not R[3] == 0:
            R[0:3] = R[0:3] / R[3]
        R_local[cell_num] = [R[0], R[1], R[2], order_n, 0, 0, R[3], \
                Numbers[0], Numbers[1], Numbers[2], Numbers[3],     \
                Numbers[4], Numbers[5], Numbers[6], Numbers[7]]
    return np.vstack((R_local, R_final))
#_____________________________________________________________________________
# Функция, рассчитывающая ускорение частицы под номером Part_num,
# полученное за счет гравитационного мультипольного взаимодействия с
# частицами в ячейке с номером cell_num.
#(Для использования в методе  Tree code)
def int_cell_to_particle(Particles, Mass_center, Part_num, cell_num):
    r_3 = m.pow(part_distance(Particles, Mass_center, Part_num, cell_num), 3)
    a_x = Mass_center[cell_num, 6] \
                * (Mass_center[cell_num, 0] - Particles[Part_num, 0]) / r_3
    a_y = Mass_center[cell_num, 6] \
                * (Mass_center[cell_num, 1] - Particles[Part_num, 1]) / r_3
    a_z = Mass_center[cell_num, 6] \
                * (Mass_center[cell_num, 2] - Particles[Part_num, 2]) / r_3
    return np.array([a_x, a_y, a_z])
#_____________________________________________________________________________
# Функция, рассчитывающая ускорение частицы под номером Part_num,
# полученное за счет гравитационного взаимодействия с частицами
# в ячейке с номером cell_num. (Для использования в методе  Tree code)
# (9.04.18)
def int_particles_to_particle(Particles, Part_num, Mass_center, cell_num):
    a_x = 0
    a_y = 0
    a_z = 0
    for num in range(int(Mass_center[cell_num, 4]), \
                     int(Mass_center[cell_num, 5])):
        if not num == Part_num:
            r_3 = m.pow(part_distance(Particles, Particles, \
                              Part_num, num), 3)
            a_x += Particles[num, 6] \
            * (Particles[num, 0] - Particles[Part_num, 0]) / r_3
            a_y += Particles[num, 6] \
            * (Particles[num, 1] - Particles[Part_num, 1]) / r_3
            a_z += Particles[num, 6] \
            * (Particles[num, 2] - Particles[Part_num, 2]) / r_3
    return np.array([a_x, a_y, a_z])
#_____________________________________________________________________________
# Функция, определяющая тип взаимодействия (частица-частица,
# мультиполь-частица), а так же раскладывающая большой куб
# на кубы меньшего размера, если удовлетворяется условие
# раскрытия ячейки (3.04.18)
#ПРОВЕРИТЬ НА ОШИБКИ
def calculate_cell(Particles, Mass_center,  \
                   Part_num, order_count,   \
                   L, cell_num):
    A = np.array([0, 0, 0])
    if not Mass_center[cell_num, 6] == 0:
        if part_distance(Particles, Mass_center, Part_num, cell_num) > L:
            A = int_cell_to_particle(Particles, Mass_center,    \
                                     Part_num, cell_num)
        else:
            if Mass_center[cell_num, 8] == 0:
                A = int_particles_to_particle(Particles, Part_num,  \
                                              Mass_center, cell_num)
            else:
                A = calculate_cube(Particles, Mass_center, Part_num,\
                                   order_count, cell_num)
    return A
#_____________________________________________________________________________
# Функция, представляющая один большой куб как 8 кубов меньшего размера
#ПРОВЕРИТЬ НА ОШИБКИ (5.04.18)
def calculate_cube(Particles, Mass_center,  \
                     Part_num, order_count, \
                     cell_num):
    n_linear = int(m.pow(2, Mass_center[cell_num, 3]))
    L = m.sqrt(3) * Distance * n / (2 * n_linear)
    order_count += m.pow(n_linear, 3)
    A = np.zeros([8, 3])
    A[0] = calculate_cell(Particles, Mass_center, Part_num, order_count, L, \
                                  int(order_count + Mass_center[cell_num, 7]))
    A[1] = calculate_cell(Particles, Mass_center, Part_num, order_count, L, \
                                  int(order_count + Mass_center[cell_num, 8]))
    A[2] = calculate_cell(Particles, Mass_center, Part_num, order_count, L, \
                                  int(order_count + Mass_center[cell_num, 9]))
    A[3] = calculate_cell(Particles, Mass_center, Part_num, order_count, L, \
                                  int(order_count + Mass_center[cell_num, 10]))
    A[4] = calculate_cell(Particles, Mass_center, Part_num, order_count, L, \
                                  int(order_count + Mass_center[cell_num, 11]))
    A[5] = calculate_cell(Particles, Mass_center, Part_num, order_count, L, \
                                  int(order_count + Mass_center[cell_num, 12]))
    A[6] = calculate_cell(Particles, Mass_center, Part_num, order_count, L, \
                                  int(order_count + Mass_center[cell_num, 13]))
    A[7] = calculate_cell(Particles, Mass_center, Part_num, order_count, L, \
                                  int(order_count + Mass_center[cell_num, 14]))
    return A.sum(axis=0)
#_____________________________________________________________________________
# Функция, подщитывающая ускорение для каждой частицы (3.04.18)
def interaction(Particles, Mass_center):
    Number_of_particles = np.size(Particles, 0)
    for Part_num in range(Number_of_particles):
        Particles[Part_num, 7:10] = G * calculate_cube(Particles, Mass_center,\
                     Part_num, 0, 0)
    return Particles
#_____________________________________________________________________________
# Функция, позволяющая получить новые параметры частиц
# из матрицы Y с помощью метода Tree code
def tree_code_gravity(Y):
    order_n = int(m.log2(n))
    Y_size = np.size(Y, 0)
#    start = time.time()
    Y = distribution(Y, Y_size)
#    computing_time = time.time() - start
#    print("Сортировка", computing_time, "с")
    Y[:, 3:6] += Y[:, 7:10] * time_step / 2
    Y[:, 0:3] += Y[:, 3:6] * time_step
#    start = time.time()
    R_final = particles_to_cell(Y, Y_size, order_n)
    while order_n > 0:
        order_n += -1
        R_final = cells_to_cell(R_final, order_n)
#    computing_time = time.time() - start
#    print("Работа с ячейками", computing_time, "с")
#    start = time.time()
    Y = interaction(Y, R_final)
#    computing_time = time.time() - start
#    print("Рассчет взаимодействия", computing_time, "с")
    Y[:, 3:6] += Y[:, 7:10] * time_step / 2
    return Y
#_____________________________________________________________________________
# Функция для "скирншота" положения всех частиц
def screenshot(System_parameters, Name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = System_parameters[:, 0]
    y = System_parameters[:, 1]
    z = System_parameters[:, 2]
    ax.scatter(x, y, z, color='red', s=0.02)
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
#X = birth_test()
X = birth_random(N)
Name = "Начальное положение.png"
screenshot(X, Name)
start = time.time()
for q in range(Steps):
    X = tree_code_gravity(X)
#    X1 = N_body_direct(X)
#    if q in [50, 100, 150, 200, 250, 300, 350, 400, 450]:
#        screenshot(X, 'Шаг' + str(q))
#print(X)
computing_time = time.time() - start
print("Полное время вычислений", computing_time, "с")
Name = "Итоговое положение.png"
screenshot(X, Name)