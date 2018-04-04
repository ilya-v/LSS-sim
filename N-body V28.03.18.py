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

#Количество ячеек по одной оси координат (для tree codes)
n = 8
# Количество частиц
N = 1000
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
# Подфункция, позволяющая сгенерировать случайные параметры для тела
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
    Sum = np.array([x, y, z, Vx, Vy, Vz, mass])
    return Sum
#_____________________________________________________________________________
# Функция, создающая i*j*k тел
def spawn_test():
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
def spawn_random(body_count):
#   Сначала создаем массив нулей, а затем заполняем его;
#   тела находятся по первому индексу, параметры - по второму
    Random_particles = np.zeros((body_count, 7))
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
#   Распределение X_size частиц по ячейкам со стороной L,
#   всего по одной оси насчитывается по n_i ячеек
def Distribution(X0, X_size, L):
    X0_width = np.size(X0, 1)
    if X0_width == 10:
        for N_local in range(X_size):
            X0[N_local, 7] = int(m.floor(X0[N_local, 0] / L))
            X0[N_local, 8] = int(m.floor(X0[N_local, 1] / L))
            X0[N_local, 9] = int(m.floor(X0[N_local, 2] / L))
    else:
        Y = np.zeros([X_size, 3])
        for N_local in range(X_size):
            Y[N_local, 0] = int(m.floor(X0[N_local, 0] / L))
            Y[N_local, 1] = int(m.floor(X0[N_local, 1] / L))
            Y[N_local, 2] = int(m.floor(X0[N_local, 2] / L))
#   Функция производит модификацию матрицы X 
#   путем добавления тройки новых координат
        X0 = np.hstack((X0, Y))
    return X0
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
def Direct_Newton(Particles, Part_num, cell_num, cell_count, Acceleration):
    a_x = 0
    a_y = 0
    a_z = 0
    for num in range(int(cell_count[cell_num, 0]), \
                     int(cell_count[cell_num, 1])):        
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
# раскрытия ячейки
#ПРОВЕРИТЬ НА ОШИБКИ
def Calculate_cell(Particles, Mass_center,      \
                   Part_num, cell_num,          \
                   oder_num, cell_count,        \
                   Acceleration):
    n_k = int(m.pow(2, Mass_center[cell_num, 7]))
    L = n * Distance / n_k 
    if not Mass_center[cell_num, 6] == 0:
        if Part_distance(Particles, Mass_center,                    \
                         Part_num, cell_num) > (L * m.sqrt(3)):
            Acceleration = Multipole_Newton(Particles, Mass_center, \
                                                Part_num, cell_num, \
                                                Acceleration)
        else:
            if L == Distance:
                Acceleration = Direct_Newton(Particles, Part_num,   \
                                             cell_num, cell_count,  \
                                             Acceleration)
            else:
                oder_num += m.pow(n_k, 3)
                Calculate_cube(Particles, Mass_center,  \
                               Part_num, cell_num,      \
                               oder_num, cell_count,    \
                               Acceleration)
    return Acceleration
#_____________________________________________________________________________
# Функция, представляющая один большой куб как 8 кубов меньшего размера
#ПРОВЕРИТЬ НА ОШИБКИ
def Calculate_cube(Particles, Mass_center,  \
                     Part_num, cell_num,    \
                     oder_num, cell_count,  \
                     Acceleration):
    n_k = int(m.pow(2, Mass_center[cell_num, 7]))
    cell_num = int(oder_num + 2 * (m.pow(n_k, 2) * Mass_center[cell_num, 3] \
                               + n_k * Mass_center[cell_num, 4]         \
                               + Mass_center[cell_num, 5]))
    cell_num_0 = cell_num
    cell_num_1 = cell_num + 1
    cell_num_2 = cell_num + n_k
    cell_num_3 = cell_num_2 + 1
    cell_num_4 = cell_num + int(m.pow(n_k, 2))
    cell_num_5 = cell_num_4 + 1
    cell_num_6 = cell_num_4 + n_k
    cell_num_7 = cell_num_6 + 1
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_0,     \
                                  oder_num, cell_count,     \
                                  Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_1,     \
                                  oder_num, cell_count,     \
                                  Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_2,     \
                                  oder_num, cell_count,     \
                                  Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_3,     \
                                  oder_num, cell_count,     \
                                  Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_4,     \
                                  oder_num, cell_count,     \
                                  Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_5,     \
                                  oder_num, cell_count,     \
                                  Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_6,     \
                                  oder_num, cell_count,     \
                                  Acceleration)
    Acceleration = Calculate_cell(Particles, Mass_center,   \
                                  Part_num, cell_num_7,     \
                                  oder_num, cell_count,     \
                                  Acceleration)
    return Acceleration
#_____________________________________________________________________________
# Функция, подщитывающая ускорение для каждой частицы
def Interaction(Particles, Mass_center, cell_count):
    Number_of_particles = np.size(Particles, 0)
    Acceleration = np.zeros([Number_of_particles, 3])
    for Part_num in range(Number_of_particles):
        Acceleration = Calculate_cube(Particles, Mass_center, \
                     Part_num, 0, 0, cell_count, Acceleration)        
    return Acceleration
#_____________________________________________________________________________
# Функция, позволяющая получить новые параметры частиц
# из матрицы Y с помощью метода Tree code
def Tree_code_gravity(Y):
    l = n * Distance / 2
    n_l = 1
    R_final = np.zeros([0, 8])
#   В цикле while идет распределение частиц по ячейкам
#   и нахождение центра масс для каждой ячейки
    while l >= Distance:
        n_i = int((n * Distance) / l)
        Y_size = np.size(Y, 0)
#       start = time.time()
        Y = Distribution(Y, Y_size, l)
#       computing_time = time.time() - start
#       print("Сортировка", computing_time, "с")
#       В Y0 идет перезапись частиц в более упорядоченном виде, чем есть в Y
        Y0 = np.zeros([Y_size, 10])
#       В R_local идут все параметры, связанные с центрами масс каждой ячейки
        R_local = np.zeros([pow(n_i, 3), 8])
        n_R_local = 0
        n2_local = 0
#       Part_in_cell -- матрица, в которой записано количество частиц 
#       в формате "номер первой частицы, номер последней частицы + 1"
        Part_in_cell = np.zeros([pow(n, 3), 2])
        n_in_cell = 0
#       Перебор всех созданных ячеек по трем координатам
#       start = time.time()
        for i in range(n_i):
            for j in range(n_i):
                for k in range(n_i):
                    R = np.zeros([4])
                    n_local = 0 
                    if n_i == n:
                        n_in_cell = int(m.pow(n, 2) * i + n * j + k)
                        Part_in_cell[n_in_cell, 0] = Part_in_cell[n_in_cell - 1, 1]
                        Part_in_cell[n_in_cell, 1] = Part_in_cell[n_in_cell, 0]
#                   Перебор всех частиц в матрице Y
                    while Y_size > 0:
                        if Y_size == n_local:
                            break
                        else:
                            if (Y[n_local, 7] == i)     \
                            and (Y[n_local, 8] == j)    \
                            and (Y[n_local, 9] == k):
                                R[0:3] += Y[n_local, 0:3] * Y[n_local, 6]
                                R[3] += Y[n_local, 6]
                                Y0[n2_local] = Y[n_local]
                                n2_local += 1
                                Y = np.delete(Y, (n_local), 0)
                                Y_size = np.size(Y, 0)
                                n_local += -1
                                if n_i == n:
                                    Part_in_cell[n_in_cell, 1] += 1
                            n_local += 1
                    if not R[3] == 0:
                        R[0:3] = R[0:3] / R[3]
                    R_local[n_R_local] = [R[0], R[1], R[2], i, j, k, R[3], n_l] 
                    n_R_local += 1
        R_final = np.vstack((R_final, R_local))
        Y = np.copy(Y0)
#       computing_time = time.time() - start
#       print("Работа с ячейками", computing_time, "с")
        n_l += 1
        l = l / 2
#   start = time.time()
    A = Interaction(Y, R_final, Part_in_cell)
#   computing_time = time.time() - start
#   print("Рассчет взаимодействия", computing_time, "с")
    Y[:, 0:3] += Y[:, 3:6] * time_step + A[:, 0:4] * m.pow(time_step, 2) / 2
    Y[:, 3:6] += A[:, 0:4] * time_step
    return Y
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
#X = spawn_cube()
X = spawn_random(N)

#start = time.time()
for q in range(Steps):
    X = Tree_code_gravity(X)
#computing_time = time.time() - start
#print("Полное время вычислений", computing_time, "с")
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
