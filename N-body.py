# -*- coding: utf-8 -*-
"""
Редактор Spyder

@author: Дмитрий Мелкозеров
"""

# v Подключаемые пакеты v
# ===========================================================================
import os
import importlib
import math as m
import time
import random as r
import numpy as np
import treecode.tree_code as tc
# import threading
from joblib import Parallel, delayed
# import statistics as stat
# import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import animation
# ===========================================================================
# ^ Подключаемые пакеты ^
# v Используемые функции v
# ===========================================================================


def parameters_test(h, p, l):
    # Подфункция, позволяющая сгенерировать определенные
    # параметры для тела
    x = Distance * (indent_i + h * period) / i_test
    y = Distance * (indent_j + p * period) / j_test
    z = Distance * (indent_k + l * period) / k_test
    # Распределение скоростей и масс считаем нормальным
    Vx = r.normalvariate(0, 4) * v_avg
    Vy = r.normalvariate(0, 4) * v_avg
    Vz = r.normalvariate(0, 4) * v_avg
    mass = abs(m_avg)
    Sum = np.array([x, y, z, Vx, Vy, Vz, mass, 0, 0, 0, 0, 0, 0, 0])
    return Sum


def randomize_parameters():
    # Подфункция, позволяющая сгенерировать случайные параметры для тела
    x = r.random() * n * Distance
    y = r.random() * n * Distance
    z = r.random() * n * Distance
#   Распределение скоростей и масс считаем нормальным
#   (пока что квадратичное отклонение выбрано наугад)
    Vx = r.normalvariate(0, 4) * v_avg
    Vy = r.normalvariate(0, 4) * v_avg
    Vz = r.normalvariate(0, 4) * v_avg
    mass = abs(r.normalvariate(m_avg, 0.5*m_avg))
    Sum = np.array([x, y, z, Vx, Vy, Vz, mass, 0, 0, 0, 0, 0, 0, 0])
    return Sum


def randomize_ellipsoid():
    # Подфункция, позволяющая сгенерировать случайные параметры для тела
    x_r = 0
    y_r = 0
    z_r = 0
    particle_not_generated = True
    while particle_not_generated:
        x_r = r.random()
        y_r = r.random()
        z_r = r.random()
        x_el = (2 * x_r - 1) / a_inp
        y_el = (2 * y_r - 1) / b_inp
        z_el = (2 * z_r - 1) / c_inp
        ellipsoid = x_el * x_el + y_el * y_el + z_el * z_el
        if ellipsoid <= 1:
            particle_not_generated = False
    center = n * Distance / 2
    x = (x_r + 0.5) * center
    y = (y_r + 0.5) * center
    z = (z_r + 0.5) * center
    d_x = x - center
    d_y = y - center
    d_z = z - center
#   Распределение скоростей и масс считаем нормальным
#   (пока что квадратичное отклонение выбрано наугад)
    Vx = r.normalvariate(0, 3) * v_avg + w_y * d_z - w_z * d_y
    Vy = r.normalvariate(0, 3) * v_avg + w_z * d_x - w_x * d_z
    Vz = r.normalvariate(0, 3) * v_avg + w_x * d_y - w_y * d_x
    mass = abs(r.normalvariate(m_avg, 0.5*m_avg))
    Sum = np.array([x, y, z, Vx, Vy, Vz, mass, 0, 0, 0, 0, 0, 0, 0])
    return Sum


def birth_test():
    # Функция, создающая i*j*k тел
    # Сначала создаем массив нулей, а затем заполняем его;
    # тела находятся по первому индексу, параметры - по второму
    test_particles = np.zeros((i_test * j_test * k_test, 14))
    Num = 0
    for l in range(k_test):
        for p in range(j_test):
            for h in range(i_test):
                test_particles[Num] = parameters_test(h, p, l)
                Num += 1
    return test_particles


def birth_random(body_count):
    # Функция, создающая "body_count" тел
    # Сначала создаем массив нулей, а затем заполняем его;
    # тела находятся по первому индексу, параметры - по второму
    random_particles = np.zeros((body_count, 14))
    for l in range(body_count):
        random_particles[l] = randomize_parameters()
    return random_particles


def birth_ellipsoid(body_count):
    # Функция, создающая "body_count" тел
    # Сначала создаем массив нулей, а затем заполняем его;
    # тела находятся по первому индексу, параметры - по второму
    random_particles = np.zeros([body_count, 14])
    for l in range(body_count):
        random_particles[l] = randomize_ellipsoid()
    return random_particles


def distribution(X0, X_size):
    # Распределение X_size частиц по ячейкам со стороной Distance
    # с последующей сортировкой по номерам ячеек (3.04.18)
    for N_local in range(X_size):
        n_x = int(m.floor(X0[N_local, 0] / Distance))
        n_y = int(m.floor(X0[N_local, 1] / Distance))
        n_z = int(m.floor(X0[N_local, 2] / Distance))
        if (n_x > n) or (n_y > n) or (n_z > n) or \
                (n_x < 0) or (n_y < 0) or (n_z < 0):
            X0[N_local, 11] = -1
        else:
            X0[N_local, 11] = n_x * n * n + n_y * n + n_z
    return X0[X0[:, 11].argsort(kind='mergesort')]


def particles_to_cell(Y, Y_size, order_n, n_max):
    # Функция, определяющая параметры самых малых ячеек из параметров
    # находящихся внутри частиц (13.04.18)
    R_local = np.zeros([n_max, 23])
    part_num = 0
    part_count = 0
    L_2 = 3 * Distance * Distance
    while Y[part_num, 11] < 0:
        part_num += 1
    for cell_num in range(n_max):
        R = np.zeros([12])
        if not part_num == Y_size:
            while Y[part_num, 11] == cell_num:
                R[0:3] += Y[part_num, 0:3] * Y[part_num, 6]
                R[3] += Y[part_num, 6]
                part_num += 1
                if part_num == Y_size:
                    break
        R[4] = part_count
        R[5] = part_num
        part_count = part_num
        d_xy = 0
        d_xz = 0
        d_yz = 0
        if not R[3] == 0:
            # Расчет положения центра масс ячейки
            R[0:3] = R[0:3] / R[3]
            # Расчет положения геометрического центра ячейки
            cell_x = cell_num // (n * n)
            R[6] = Distance * (0.5 + cell_x)
            R[7] = Distance * (0.5 + ((cell_num // n) - cell_x * n))
            R[8] = Distance * (0.5 + (cell_num % n))
            # Расчет квадрупольного момента для выбранной ячейки
            for s in range(int(R[4]), int(R[5])):
                R[9] += Y[s, 6] * (Y[s, 0] - R[0]) * (Y[s, 1] - R[1])
                R[10] += Y[s, 6] * (Y[s, 0] - R[0]) * (Y[s, 2] - R[2])
                R[11] += Y[s, 6] * (Y[s, 1] - R[1]) * (Y[s, 2] - R[2])
                d_xy += Y[s, 6] * Y[s, 0] * Y[s, 1]
                d_xz += Y[s, 6] * Y[s, 0] * Y[s, 2]
                d_yz += Y[s, 6] * Y[s, 1] * Y[s, 2]
            R[9:12] *= 3
            # Итоговый вид строки с параметрами ячейки
        R_local[cell_num] = [R[0], R[1], R[2], R[6], R[7], R[8],
                             R[3], R[9], R[10], R[11], L_2, order_n,
                             R[4], R[5], 0, 0, 0, 0, 0, 0,
                             d_xy, d_xz, d_yz]
    return R_local


def cells_to_cell(R_final, order_n, n_max):
    # Функция, вычисляющая параметры ячеек за счет
    # находящихся внутри ячеек с меньшим порядком (13.04.18)
    cell_length = Distance * (n / order_n)
    n_linear = order_n * 2
    n_total = int(m.pow(order_n, 3))
    R_local = np.zeros([n_total, 23])
    L_2 = 3 * Distance * Distance * n * n / (order_n * order_n)
    for cell_num in range(n_total):
        R = np.zeros([10])
        cell_x = cell_num // (order_n * order_n)
        cell_y = (cell_num // order_n) - cell_x * order_n
        cell_z = cell_num % order_n
        cell_num_0 = 2 * int(cell_x * n_linear * n_linear
                             + cell_y * n_linear + cell_z)
        Numbers = [cell_num_0, cell_num_0 + 1,
                   cell_num_0 + int(n_linear),
                   cell_num_0 + int(n_linear) + 1,
                   cell_num_0 + int(n_linear * n_linear),
                   cell_num_0 + int(n_linear * n_linear) + 1,
                   cell_num_0 + int(n_linear * n_linear + n_linear),
                   cell_num_0 + int(n_linear * n_linear + n_linear) + 1]
        d_xy = 0
        d_xz = 0
        d_yz = 0
#        D_xy = 0
#        D_xz = 0
#        D_yz = 0
        for u in range(8):
            # Определяем параметры центра масс
            R[0:3] += R_final[Numbers[u], 0:3] \
                    * R_final[Numbers[u], 6]
            R[3] += R_final[Numbers[u], 6]
            # Определяем доп. параметры, связанные с квадрупольным вкладом
#            D_xy += R_final[Numbers[u], 6]  \
#                * R_final[Numbers[u], 0] * R_final[Numbers[u], 1]
#            D_xz += R_final[Numbers[u], 6]  \
#                * R_final[Numbers[u], 0] * R_final[Numbers[u], 2]
#            D_yz += R_final[Numbers[u], 6]  \
#                * R_final[Numbers[u], 1] * R_final[Numbers[u], 2]
#            d_xy += R_final[Numbers[u], 20]
#            d_xz += R_final[Numbers[u], 21]
#            d_yz += R_final[Numbers[u], 22]
        if not R[3] == 0:
            # Расчет положения ЦМ и геометрического центра ячейки
            R[0:3] = R[0:3] / R[3]
            R[4] = cell_length * (0.5 + cell_x)
            R[5] = cell_length * (0.5 + cell_y)
            R[6] = cell_length * (0.5 + cell_z)
            # Расчет квадрупольного момента для выбранной ячейки
#            for s in range(8):
#                if not R_final[Numbers[s], 6] == 0:
#                    R[7] += R_final[Numbers[s], 6]          \
#                        * (R_final[Numbers[s], 0] - R[0])   \
#                        * (R_final[Numbers[s], 1] - R[1])
#                    R[8] += R_final[Numbers[s], 6]          \
#                        * (R_final[Numbers[s], 0] - R[0])   \
#                        * (R_final[Numbers[s], 2] - R[2])
#                    R[9] += R_final[Numbers[s], 6]          \
#                        * (R_final[Numbers[s], 1] - R[1])   \
#                        * (R_final[Numbers[s], 2] - R[2])
#            if (R[7] == 0) and (R[8] == 0) and (R[9] == 0):
#                R[7] = R_final[Numbers[:], 7].sum()
#                R[8] = R_final[Numbers[:], 8].sum()
#                R[9] = R_final[Numbers[:], 9].sum()
#            else:
#                R[7] += d_xy - D_xy
#                R[8] += d_xz - D_xz
#                R[9] += d_yz - D_yz
#                R[7:10] *= 3
#        Итоговый вид строки с параметрами ячейки
        R_local[cell_num] = [R[0], R[1], R[2], R[4], R[5], R[6], R[3],
                             R[7], R[8], R[9], L_2, order_n,
                             Numbers[0], Numbers[1], Numbers[2], Numbers[3],
                             Numbers[4], Numbers[5], Numbers[6], Numbers[7],
                             d_xy, d_xz, d_yz]
#    Корректируем номера "дочерних" ячеек
    R_local[:, 12:20] += n_total
    R_final[0:(-n_max), 12:20] += n_total
    return np.vstack((R_local, R_final))


def tree_root(Particles, Mass_center):
    # Функция, с которой начинается tree code
    if use_multiprocessing:
        A0 = Parallel(n_jobs=workers, verbose=0)(
                delayed(tc.begin_tree)(Particles, Mass_center, i,
                                       n, eps_smooth)
                for i in range(1, 9))
        A = A0[0] + A0[1] + A0[2] + A0[3] + A0[4] + A0[5] + A0[6] + A0[7]
    else:
        A = np.zeros([np.size(Particles, 0), 4])
        if not Mass_center[1, 6] == 0:
            A += tc.begin_tree(Particles, Mass_center, 1, n, eps_smooth)
        if not Mass_center[2, 6] == 0:
            A += tc.begin_tree(Particles, Mass_center, 2, n, eps_smooth)
        if not Mass_center[3, 6] == 0:
            A += tc.begin_tree(Particles, Mass_center, 3, n, eps_smooth)
        if not Mass_center[4, 6] == 0:
            A += tc.begin_tree(Particles, Mass_center, 4, n, eps_smooth)
        if not Mass_center[5, 6] == 0:
            A += tc.begin_tree(Particles, Mass_center, 5, n, eps_smooth)
        if not Mass_center[6, 6] == 0:
            A += tc.begin_tree(Particles, Mass_center, 6, n, eps_smooth)
        if not Mass_center[7, 6] == 0:
            A += tc.begin_tree(Particles, Mass_center, 7, n, eps_smooth)
        if not Mass_center[8, 6] == 0:
            A += tc.begin_tree(Particles, Mass_center, 8, n, eps_smooth)
    return A


def tree_code_gravity(Y):
    # Функция, позволяющая получить новые параметры частиц
    # из матрицы Y с помощью метода Tree code (13.04.18)
    order_n = n
    Y_size = np.size(Y, 0)
    Y[:, 3:6] += Y[:, 7:10] * time_step / 2
    Y[:, 0:3] += Y[:, 3:6] * time_step
    Y = distribution(Y, Y_size)
    n_max = int(n * n * n)
    R_final = particles_to_cell(Y, Y_size, order_n, n_max)
    while order_n > 1:
        order_n *= 0.5
        R_final = cells_to_cell(R_final, order_n, n_max)
    Y[:, 7:11] = tree_root(Y, R_final)
    if Y[0, 11] < 0:
        Y = tc.N_body_direct(Y)
    Y[:, 7:11] *= G
    Y[:, 3:6] += Y[:, 7:10] * time_step / 2
    return Y


def momentum_of_system(Y):
    # Функция, определяющая импульс всей системы и выводящая его в строку
    P = np.zeros([np.size(Y, 0), 3])
    P[:, 0] = np.multiply(Y[:, 3], Y[:, 6])
    P[:, 1] = np.multiply(Y[:, 4], Y[:, 6])
    P[:, 2] = np.multiply(Y[:, 5], Y[:, 6])
    print('Полный импульс системы ', P.sum(axis=0))


def momentum_of_particles(Y):
    # Функция, определяющая импульс всех материальных точек
    P = np.zeros([np.size(Y, 0), 3])
    P[:, 0] = np.multiply(Y[:, 3], Y[:, 6])
    P[:, 1] = np.multiply(Y[:, 4], Y[:, 6])
    P[:, 2] = np.multiply(Y[:, 5], Y[:, 6])
    if np.size(Y, 0) > 10:
        print('Импульсы всех материальных точек сохранены в файл')
        np.savetxt('Импульсы материальных точек.txt', P)
    else:
        print(P)


def kinetic_energy_Newton(Y):
    # Функция, определяющая кинетическую энергию каждой частицы
    V = np.multiply(Y[:, 3:6], Y[:, 3:6])
    E = V.sum(axis=1)
    E = np.multiply(E[:], Y[:, 6])
    E /= 2
    return E


def max_dT(Y):
    # Функция, определяющая максимальную разницу
    # кинетической энергии частиц за шаг
    E = kinetic_energy_Newton(Y)
    E = E - Y[:, 12]
    dE_plus = np.amax(E)
    dE_minus = np.amin(E)
    if abs(dE_minus) > dE_plus:
        dE = dE_minus
    else:
        dE = dE_plus
    return dE


def max_dU(Y):
    # Функция, определяющая максимальную разницу
    # потенциальной энергии частиц за шаг
    E = potential_energy_Newton(Y)
    E = E - Y[:, 13]
    dE_plus = np.amax(E)
    dE_minus = np.amin(E)
    if abs(dE_minus) > dE_plus:
        dE = dE_minus
    else:
        dE = dE_plus
    return dE


def plot_max_dE_kinetic(dE):
    # Функция, создающая график максимальной разницы
    # кинетической энергии частиц за все время работы программы
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dE[1:, 0], dE[1:, 4])
    ax.set_xlabel('Номер шага')
    ax.set_ylabel('Kinetic energy')
    ax.set_title('Max kinetic energy difference per step')
    plt.savefig('Максимальное изменение кинетической энергии за шаг', dpi=640)
    plt.show()


def plot_max_dE_potential(dE):
    # Функция, создающая график максимальной разницы
    # потенциальной энергии частиц за все время работы программы
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dE[1:, 0], dE[1:, 5])
    ax.set_xlabel('Номер шага')
    ax.set_ylabel('Potential energy')
    ax.set_title('Max potential energy difference per step')
    plt.savefig('Максимальное изменение потенциальной энергии за шаг', dpi=640)
    plt.show()


def potential_energy_Newton(Y):
    # Функция, определяющая кинетическую энергию каждой частицы
    E = np.multiply(Y[:, 10], Y[:, 6])
    return E


def system_kinetic_energy(Y):
    # Функция, определяющая полную энергию системы
    E = kinetic_energy_Newton(Y)
    E = E.sum(axis=0)
    return E


def system_potential_energy(Y):
    E = potential_energy_Newton(Y)
    E = E.sum(axis=0)
    return E


def system_energy_Newton(Y):
    # Функция, определяющая полную энергию системы
    E = system_kinetic_energy(Y)
    E = E + system_potential_energy(Y)
    return E


def plot_avg(E):
    # Функция, создающая график кинетической энергии частиц
    # за все время работы программы
    Energy = np.copy(E[:, 1:3])
    Energy /= N
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax.plot(E[:, 0], Energy[:, 0])
    ax1.plot(E[:, 0], Energy[:, 1])
    ax.xaxis.set_ticklabels([])
    ax1.set_xlabel('Номер шага')
    ax.set_ylabel('Kinetic enegry')
    ax1.set_ylabel('Potential energy')
    ax.set_title('Average energy')
    ax1.set_title(' ')
    plt.savefig('Средняя энергия материальной точки', dpi=640)
    plt.show()


def plot_system_enegry(E):
    # Функция, создающая график потенциальной энергии частиц
    # за все время работы программы
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax.plot(E[:, 0], E[:, 1])
    ax1.plot(E[:, 0], Energy[:, 2])
    ax.xaxis.set_ticklabels([])
    ax1.set_xlabel('Номер шага')
    ax.set_ylabel('Kinetic enegry')
    ax1.set_ylabel('Potential energy')
    ax.set_title('Energy at step')
    plt.savefig('Кинетическая и потенциальная энергия системы', dpi=640)
    plt.show()


def plot_total_energy(E):
    # Функция, создающая график потенциальной энергии частиц
    # за все время работы программы
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(E[:, 0], E[:, 3])
    ax.set_xlabel('Номер шага')
    ax.set_ylabel('Энергия')
    ax.set_title('Полная энергия системы')
    plt.savefig('Полная энергия системы', dpi=640)
    plt.show()


def plot_combined_energy(E):
    # Функция, создающая график потенциальной энергии частиц
    # за все время работы программы
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(E[:, 0], E[:, 3], label='Полная энергия', color='black')
    ax.plot(E[:, 0], E[:, 1], label='Кинетическая энергия', color='red')
    ax.plot(E[:, 0], E[:, 2], label='Потенциальная энергия', color='blue')
    ax.set_xlabel('Номер шага')
    ax.set_ylabel('Энергия')
    ax.set_title('Полная энергия системы')
    plt.legend()
    plt.savefig('Кинетическая, потенциальная, полная энергия системы', dpi=640)
    plt.show()


def is_gravity_field_weak(Y):
    # Функция, выдающая ошибку, если гравитационное поле становится
    # слишком сильным для применения используемой модели
    global error
    global error_name
    Array_phi = abs(Y[:, 10] / c_2)
    Array_phi = Array_phi >= 0.05
    if Array_phi.any():
        error = True
        error_name = 'Strong gravity field error'


def speed_limit(Y):
    # Функция, выдающая ошибку если скорость материальной
    # точки станет больше скорости света
    global error
    global error_name
    V = np.zeros([np.size(Y, 0), 3])
    V = np.multiply(Y[:, 3:6], Y[:, 3:6])
    V_2 = V.sum(axis=1) >= c_2
    if V_2.any():
        error = True
        error_name = 'FTL error'


def screenshot(System_parameters, name, point_size):
    # Функция для "скирншота" положения всех частиц
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = System_parameters[:, 0]
    y = System_parameters[:, 1]
    z = System_parameters[:, 2]
    ax.scatter(x, y, z, color='red', s=point_size)
    ax.autoscale(False)
    ax.set_xlabel('x, кпк')
    ax.set_ylabel('y, кпк')
    ax.set_zlabel('z, кпк')
    plt.savefig(name, dpi=1280)
#    plt.show()


def input_int_value(msg_0, msg_1, msg_2):
    print(msg_0)
    continue_input = True
    while continue_input:
        try:
            variable = int(input())
            if variable > 0:
                continue_input = False
        except ValueError:
            print(msg_1)
            print(msg_2)
    return variable


def input_float_value(msg_0, msg_00, msg_000, msg_1, msg_2):
    print(msg_0)
    print(msg_00)
    print(msg_000)
    continue_input = True
    while continue_input:
        try:
            variable = float(input())
            if variable >= 0:
                continue_input = False
            else:
                print('Введено некорректное значение. Попробуйте еще раз')
        except ValueError:
            print(msg_1)
            print(msg_2)
    return variable


def input_float_less_1_value(msg_0, msg_00, msg_1, crit):
    print(msg_0)
    print(msg_00 + str(crit))
    continue_input = True
    while continue_input:
        try:
            variable = float(input())
            if (variable >= -1) and (variable <= 1):
                continue_input = False
            else:
                print('Введено некорректное значение. Попробуйте еще раз')
        except ValueError:
            print(msg_1)
    return variable

# ===========================================================================
# ^ Используемые функции ^


if __name__ == "__main__":
    importlib.reload(tc)
    # v Константы v
    # =======================================================================
    # Гравитационная постоянная
    # G = 6.67408313 * m.pow(10, -11)  # м^3/(кг*с^2)
    G = 4.51811511 * m.pow(10, -15)  # кпк^3/(М_(Солнца)* (10^12 с)^2)
    # G = 4.51811511 * m.pow(10, -7)  # кпк^3/(М_(Млечного пути)* (10^15 с)^2)
# Скорость света
    # c = 299792458 # м/с
    c = 9.7156188999  # кпк/(10^12 с)
# ===========================================================================
# ^ Константы ^
# v Параметры системы v
# ===========================================================================
# Прочие переменные (желательно не трогать)
    marker_size = 0.2  # 1
    c_2 = c * c
    error = False
    error_name = ''
    not_forbid_launch = True
    continue_input = True
    interrupted = False
    workers = os.cpu_count()
    msg_N_0 = 'Введите число материальных точек'
    msg_N_1 = 'Число материальных точек всегда должно быть целым'
    msg_N_2 = 'Введите число материальных точек еще раз'
    msg_n_0 = 'Введите количество ячеек в формате 2^n (нужно задать n)'
    msg_n_1 = 'Число ячеек всегда должно быть целым'
    msg_n_2 = 'Введите число ячеек еще раз'
    msg_steps_0 = 'Введите число временных шагов'
    msg_steps_1 = 'Введено недопустимое число шагов'
    msg_steps_2 = 'Введите число шагов еще раз'
    msg_m_0 = 'Введите среднюю массу материальных точкек в массах галактик'
    msg_m_00 = '(Масса галактики имеет порядок 10^41 кг)'
    msg_m_1 = 'Cредняя масса материальной точки должна быть числом'
    msg_m_2 = 'Введите среднюю массу еще раз'
    msg_v_0 = 'Введите среднюю скорость материальных точкек в кпк/(10^12 с)'
    msg_v_00 = '(1 кпк/(10^12 с) = 3,08567758*10^7 м/с)'
    msg_v_000 = 'ВАЖНО ПОМНИТЬ! c = 9.7156188999 кпк/(10^12 с)'
    msg_v_1 = 'Cредняя скорость материальной точки должна быть числом'
    msg_v_2 = 'Введите среднюю скорость материальных точек еще раз'
    msg_d_0 = 'Введите размер ячейки в кпк'
    msg_d_1 = 'Размер ячейки должен быть в виде числа'
    msg_d_2 = 'Введите размер ячейки еще раз'
    msg_t_0 = 'Введите временной шаг в единицах (10^12 с)'
    msg_t_1 = 'Временной шаг должен быть в виде числа'
    msg_t_2 = 'Введите временной шаг еще раз'
    msg_ind_0 = 'Введите отступ от границы рассматриваемой'
    msg_ind_i_0 = 'области по оси X в кпк'
    msg_ind_j_0 = 'области по оси Y в кпк'
    msg_ind_k_0 = 'области по оси Z в кпк'
    msg_ind_1 = 'Отступ должен быть в виде числа'
    msg_ind_2 = 'Введите отступ еще раз'
    msg_i_0 = 'Введите число материальных точек по оси X'
    msg_j_0 = 'Введите число материальных точек по оси Y'
    msg_k_0 = 'Введите число материальных точек по оси Z'
    msg_axis_1 = 'Число материальных точек всегда должно быть целым'
    msg_axis_2 = 'Введите число материальных точек еще раз'
    msg_per_0 = 'Введите расстояние между двумя соседними точками,'
    msg_per_00 = 'расположенных на одной оси в единицах длины ячейки'
    msg_per_1 = 'Расстояние должно быть в виде числа'
    msg_per_2 = 'Введите расстояние ячейки еще раз'
    msg_a_0 = 'Введите величину полуоси эллипсоида по оси X'
    msg_b_0 = 'Введите величину полуоси эллипсоида по оси Y'
    msg_c_0 = 'Введите величину полуоси эллипсоида по оси Z'
    msg_abc_0 = 'от 0 до 1. Где 1 соответствует четверти размера системы'
    msg_abc_1 = 'Длина полуоси должна быть числом'
    msg_w_0 = 'Введите начальную угловую скорость в размерности рад/(10^12 с)'
    msg_wx_0 = 'в плоскости YZ. Величина не должна превышать '
    msg_wy_0 = 'в плоскости XZ. Величина не должна превышать '
    msg_wz_0 = 'в плоскости XY. Величина не должна превышать '
    msg_w_1 = 'Угловая скорость должна быть числом'
    msg_eps_0 = 'Введите смягчающую длину потенциала в кпк'
    msg_eps_1 = 'Смягчающая длина должна быть числом'

# Временной интервал
    # time_step = pow(10, 13)  # с
    time_step = 100.0  # 0.000025  # 10^12 с
    # time_step = 0.01  # 10^15 с

# Процентное распределение материи по типу
    d_e = 0.70  # Темная энергия
    d_m = 0.25  # Темная материя
    v_m = 0.05  # Видимая материя

# Параметр "сглаживания" гравитационного взаимодействия на близких дистанциях
    eps_smooth = 5.0  # кпк

# Параметры, которые нужны чаще всего (можно и нужно трогать)
# Количество ячеек по одной оси координат (для tree codes) в виде 2^(n)
    n = 4

# Минимальный размер ячейки по одной оси координат
    # Distance = 2 * 3.08567758 * pow(10, 22) # м
    Distance = 10 * m.pow(10, 3)  # кпк
    # Distance = 5 # Мпк

# Задаем первоначальный размер системы в единицах "Distance"
# для функции parameters_test
    i_test = 10
    j_test = 10
    k_test = 10
    indent_i = 0.0
    indent_j = 0.0
    indent_k = 0.0

# Параметры генерации эллипсоида в единицах (n * Distance / 2)
    a_inp = 1.0
    b_inp = 1.0
    c_inp = 1.0
# Начальные угловые скорости эллипсоида
    w_x = 0.0
    w_y = 0.0
    w_z = 0.0000005

# Средняя масса наблюдаемых объектов и их пекулярная скорость
    # m_avg = 1.98892 * pow(10, 41) # кг
    # v_avg = 0 #4 * pow(10, 5) / np.sqrt(3) # м/с
    m_avg = pow(10, 11)  # масс Солнц
    v_avg = 0.0  # 1.3 * pow(10, -2) / np.sqrt(3) # кпк/(10^12 c)
#     m_avg = 1 #масс Млечного пути
    # v_avg = 0 #1.3 * pow(10, -2) / np.sqrt(3) # Мпк/(10^15 c)

# Количество частиц
    N = 1000
# Число шагов
    Steps = 1
# Номера шагов, на которых требуется "сфотографировать положение всех
# материальных точек
    make_prelaunch_screenshot = False
    scr_step = [1000, 1500, 2000, 2500, 3000, 3500, 4000,
                4500, 5000, 5500, 6000, 6500, 7000, 7500,
                8000, 8500, 9000, 9500, 10000, 10500, 11000,
                11500, 12000, 12500, 13000, 13500, 14000,
                14500, 15000]
# Тип сгенерированной системы (обязательно заполнить!)
    system_generation_type = 'last'
# Использовать несколько процессов для вычислений
    use_multiprocessing = False
# Использовать данные, введенные вручную
    use_manual_input = True
# Использовать телеметрию
    use_telemetry = False
# Обратить время вспять
    inverse_time = False
# ===========================================================================
# ^ Параметры системы ^
# v Область с исполняемым кодом v
# ===========================================================================
    if use_manual_input:
        print('Введите название используемой конфигурации системы')
        system_generation_type = str(input())
        Distance = input_float_value(msg_d_0, '', '', msg_d_1, msg_d_2)
        n = input_int_value(msg_n_0, msg_n_1, msg_n_2)
        time_step = input_float_value(msg_t_0, '', '', msg_t_1, msg_t_2)
        Steps = input_int_value(msg_steps_0, msg_steps_1, msg_steps_2)
        eps_smooth = input_float_value(msg_eps_0, '', '', msg_eps_1, '')
        if (system_generation_type == 'random') or \
                (system_generation_type == 'cube') or\
                (system_generation_type == 'ellipsoid'):
            m_avg = input_float_value(msg_m_0, msg_m_00, '', msg_m_1, msg_m_2)
            m_avg *= m.pow(10, 11)
            v_avg = input_float_value(msg_v_0, msg_v_00, msg_v_000,
                                      msg_v_1, msg_v_2)
            if system_generation_type == 'random':
                N = input_int_value(msg_N_0, msg_N_1, msg_N_2)
            if system_generation_type == 'ellipsoid':
                N = input_int_value(msg_N_0, msg_N_1, msg_N_2)
                w_crit = 2 * c / (n * Distance)
                print(w_crit)
                a_inp = input_float_less_1_value(msg_a_0, msg_abc_0,
                                                 msg_abc_1, '')
                b_inp = input_float_less_1_value(msg_b_0, msg_abc_0,
                                                 msg_abc_1, '')
                c_inp = input_float_less_1_value(msg_c_0, msg_abc_0,
                                                 msg_abc_1, '')
                w_x = input_float_less_1_value(msg_w_0, msg_wx_0,
                                               msg_w_1, w_crit)
                w_y = input_float_less_1_value(msg_w_0, msg_wy_0,
                                               msg_w_1, w_crit)
                w_z = input_float_less_1_value(msg_w_0, msg_wz_0,
                                               msg_w_1, w_crit)
        print('Делать скриншоты системы?')
        print('y/n')
        input_variable = input()
        if (input_variable == 'y') or (input_variable == 'n'):
            if input_variable == 'y':
                make_prelaunch_screenshot = True
                enable_screenshots = True
                print('Наберите номера шагов, на которых нужно')
                print('сделать снимок системы')
                print('После того, как все нужные номера введены,')
                print('наберите "end" без кавычек, чтобы продолжить')
                while enable_screenshots:
                    input_var = input()
                    if input_var == 'end':
                        enable_screenshots = False
                    else:
                        try:
                            temp_var = int(input_var)
                            scr_step.append(temp_var)
                        except ValueError:
                            print('Номер шага может быть только целым числом')
            else:
                make_prelaunch_screenshot = False
                scr_step = []
        else:
            print('Введено недопустимое значение')
            print('Создание скриншотов отменено')
            make_prelaunch_screenshot = False
        print('Использовать телеметрию?')
        print('y/n')
        input_variable = input()
        if (input_variable == 'y') or (input_variable == 'n'):
            use_telemetry = input_variable == 'y'
        else:
            print('Введено недопустимое значение')
            print('Телеметрия не используется')
            use_telemetry = False
        print('Использовать многоядерность?')
        print('y/n')
        input_variable = input()
        if (input_variable == 'y') or (input_variable == 'n'):
            use_multiprocessing = input_variable == 'y'
        else:
            print('Введено недопустимое значение')
            print('Многоядерность не используется')
        print('Изменить знак у временного интервала?')
        print('y/n')
        input_variable = input()
        if (input_variable == 'y') or (input_variable == 'n'):
            inverse_time = input_variable == 'y'
        else:
            print('Введено недопустимое значение')
            inverse_time = False
    if (d_e >= 0) and (d_m >= 0) and (v_m > 0) \
            and (abs(1 - d_e - d_m - v_m) < 0.00000000001):
        m_avg = m_avg * (1 + (d_m / v_m))
    else:
        not_forbid_launch = False
        print('Недопустимое соотношение типов материи')
    if (time_step <= 0) or (Distance <= 0):
        not_forbid_launch = False
        print('Недопустимые параметры системы')
    if n > 0:
        n = int(m.pow(2, int(n)))
    else:
        not_forbid_launch = False
        print('Количество ячеек не может быть нулевым или отрицательным')
    if inverse_time:
        time_step *= -1
    try:
        try:
            try:
                if system_generation_type == 'cube':
                    if use_manual_input:
                        indent_i = input_float_value(msg_ind_0, msg_ind_i_0,
                                                     '', msg_ind_1, msg_ind_2)
                        indent_j = input_float_value(msg_ind_0, msg_ind_j_0,
                                                     '', msg_ind_1, msg_ind_2)
                        indent_k = input_float_value(msg_ind_0, msg_ind_k_0,
                                                     '', msg_ind_1, msg_ind_2)
                        i_test = input_int_value(msg_i_0, msg_axis_1,
                                                 msg_axis_2)
                        j_test = input_int_value(msg_j_0, msg_axis_1,
                                                 msg_axis_2)
                        k_test = input_int_value(msg_k_0, msg_axis_1,
                                                 msg_axis_2)
                        period = input_float_value(msg_per_0, msg_per_00, '',
                                                   msg_per_1, msg_per_2)
                    X = birth_test()
                    np.savetxt('last config.txt', X)
                elif system_generation_type == 'random':
                    X = birth_random(N)
                    np.savetxt('last config.txt', X)
                elif system_generation_type == 'ellipsoid':
                    if (a_inp == 0) or (b_inp == 0) or (c_inp == 0):
                        not_forbid_launch = False
                        print('Полуоси эллипсоида не могут быть нулевыми')
                    else:
                        X = birth_ellipsoid(N)
                        np.savetxt('last config.txt', X)
                elif system_generation_type == 'last':
                    X = np.loadtxt('last config.txt', dtype='float64')
                elif system_generation_type == 'debug':
                    X = np.loadtxt('error config.txt', dtype='float64')
                elif system_generation_type == 'test':
                    X = np.loadtxt('test config.txt', dtype='float64')
                elif system_generation_type == 'final':
                    X = np.loadtxt('final config.txt', dtype='float64')
                else:
                    not_forbid_launch = False
                    print('Выбранная конфигурация не может быть загружена')
            except IOError:
                not_forbid_launch = False
                print('Отсутствует необходимый файл конфигурации')
        except TypeError:
            not_forbid_launch = False
            print('Число материальных точек всегда должно быть целым')
    except ValueError:
        not_forbid_launch = False
        print('Неприемлимое число материальных точек')
    if not_forbid_launch:
        if np.size(X, 1) == 12:
            migration = np.zeros([np.size(X, 0), 2])
            X = np.hstack((X, migration))
            np.savetxt('last config.txt', X)
        if workers >= 8:
            workers = 8
        elif workers >= 4:
            workers = 4
        elif workers >= 2:
            workers = 2
        else:
            use_multiprocessing = False
        try:
            if make_prelaunch_screenshot:
                screenshot(X, 'Шаг 0', marker_size)
            Energy = np.zeros([Steps, 6])
            start = time.time()
            for q in range(Steps):
                speed_limit(X)
                is_gravity_field_weak(X)
                if error:
                    np.savetxt('error config.txt', X)
                    screenshot(X, error_name, marker_size)
                    print(error_name + ' at step ' + str(q))
                    break
                X = tree_code_gravity(X)
                Energy[q] = [q,
                             system_kinetic_energy(X),
                             system_potential_energy(X),
                             system_energy_Newton(X),
                             max_dT(X),
                             max_dU(X)]
                X[:, 12] = kinetic_energy_Newton(X)
                X[:, 13] = potential_energy_Newton(X)
                if q in scr_step:
                    screenshot(X, 'Шаг ' + str(q), marker_size)
            computing_time = time.time() - start
            print("Время выполнения", computing_time, "с")
            if use_telemetry:
                momentum_of_system(X)
                plot_max_dE_kinetic(Energy)
                plot_max_dE_potential(Energy)
                plot_avg(Energy)
                plot_system_enegry(Energy)
                plot_total_energy(Energy)
                plot_combined_energy(Energy)
        except KeyboardInterrupt:
            print('Работа программы прервана')
            momentum_of_system(X)
            plot_max_dE_kinetic(Energy)
            plot_max_dE_potential(Energy)
            plot_avg(Energy)
            plot_system_enegry(Energy)
            plot_total_energy(Energy)
            plot_combined_energy(Energy)
        print('Сохранить финальную конфигурацию системы?')
        print('y/n')
        input_variable = input()
        if input_variable == 'y':
            np.savetxt('final config.txt', X)
        elif input_variable == 'n':
            print('Конфигурация не будет сохранена')
        else:
            print('Введено недопустимое значение')
# ===========================================================================
# ^ Область с исполняемым кодом ^
