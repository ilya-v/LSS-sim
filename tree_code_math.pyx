import numpy as np
from numpy cimport ndarray
cimport numpy as np
from libc.math cimport sqrt
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def part_distance(np.ndarray[DTYPE_t, ndim=2] Particle_1,
                  np.ndarray[DTYPE_t, ndim=2] Particle_2,
                  int Number_1, int Nubmer_2):
    assert Particle_1.dtype == DTYPE and Particle_2.dtype == DTYPE
    # Функция, которая выдает растояние между частицам 1 и 2
    cdef double delta_x = Particle_1[Number_1, 0] - Particle_2[Nubmer_2, 0]
    cdef double delta_y = Particle_1[Number_1, 1] - Particle_2[Nubmer_2, 1]
    cdef double delta_z = Particle_1[Number_1, 2] - Particle_2[Nubmer_2, 2]
    cdef double r = sqrt(delta_x * delta_x
                         + delta_y * delta_y
                         + delta_z * delta_z)
    return r


def smooth_distance(np.ndarray[DTYPE_t, ndim=2] Particles,
                    int Num_1, int Num_2, double eps_smooth):
    assert Particles.dtype == DTYPE
    # Функция, выдающая растояние между частицам 1 и 2
    cdef double delta_x = Particles[Num_1, 0] - Particles[Num_2, 0]
    cdef double delta_y = Particles[Num_1, 1] - Particles[Num_2, 1]
    cdef double delta_z = Particles[Num_1, 2] - Particles[Num_2, 2]
    cdef double delta_soft = sqrt(delta_x * delta_x
                                  + delta_y * delta_y
                                  + delta_z * delta_z
                                  + eps_smooth * eps_smooth)
    return delta_soft  # * delta_soft * delta_soft


def int_Ps_to_P(np.ndarray[DTYPE_t, ndim=2] Particles,
                int n1, int n2,
                int n_1, int n_2, double eps_smooth):
    # Функция, рассчитывающая ускорение частицы под номером Part_num,
    # полученное за счет гравитационного взаимодействия с частицами
    # в ячейке с номером cell_num. (Для использования в методе  Tree code)
    cdef double a_x
    cdef double a_y
    cdef double a_z
    cdef double phi
    cdef double r_1
    cdef double r_3
    cdef int part_num
    cdef int num
    cdef int counter = 0
    cdef ndarray A = np.zeros([n2 - n1, 4], dtype=DTYPE)
    for part_num in range(n1, n2):
        a_x = 0
        a_y = 0
        a_z = 0
        phi = 0
        if (part_num >= n_1) and (part_num < n_2):
            for num in range(n_1, n_2):
                if not num == part_num:
                    r_1 = smooth_distance(Particles, part_num, num, eps_smooth)
                    r_3 = r_1 * r_1 * r_1
                    a_x += Particles[num, 6] \
                        * (Particles[num, 0] - Particles[part_num, 0]) / r_3
                    a_y += Particles[num, 6] \
                        * (Particles[num, 1] - Particles[part_num, 1]) / r_3
                    a_z += Particles[num, 6] \
                        * (Particles[num, 2] - Particles[part_num, 2]) / r_3
                    phi += Particles[num, 6] / r_1
        else:
            for num in range(n_1, n_2):
                r_1 = smooth_distance(Particles, part_num, num, eps_smooth)
                r_3 = r_1 * r_1 * r_1
                a_x += Particles[num, 6] \
                    * (Particles[num, 0] - Particles[part_num, 0]) / r_3
                a_y += Particles[num, 6] \
                    * (Particles[num, 1] - Particles[part_num, 1]) / r_3
                a_z += Particles[num, 6] \
                    * (Particles[num, 2] - Particles[part_num, 2]) / r_3
                phi += Particles[num, 6] / r_1
        A[counter, 0] = a_x
        A[counter, 1] = a_y
        A[counter, 2] = a_z
        A[counter, 3] = - phi
        counter += 1
    return A


def int_C_to_P(np.ndarray[DTYPE_t, ndim=2] Particles,
               np.ndarray[DTYPE_t, ndim=2] Mass_center,
               int Part_num, int cell_num):
    # Функция, рассчитывающая ускорение частицы под номером Part_num,
    # полученное за счет гравитационного мультипольного взаимодействия с
    # частицами в ячейке с номером cell_num.
    cdef ndarray A = np.zeros([4])
    cdef double delta_x = Mass_center[cell_num, 0] - Particles[Part_num, 0]
    cdef double delta_y = Mass_center[cell_num, 1] - Particles[Part_num, 1]
    cdef double delta_z = Mass_center[cell_num, 2] - Particles[Part_num, 2]
    cdef double r_1 = part_distance(Particles, Mass_center,
                                    Part_num, cell_num)
    cdef double r_3 = r_1 * r_1 * r_1
    delta_x *= Mass_center[cell_num, 6] / r_3
    delta_y *= Mass_center[cell_num, 6] / r_3
    delta_z *= Mass_center[cell_num, 6] / r_3
    cdef double phi = - Mass_center[cell_num, 6] / r_1
#    cell_to_body += quadrupole(Mass_center, cell_num, r_1, r_3,
#                               delta_x, delta_y, delta_z)
    A[0] = delta_x
    A[1] = delta_y
    A[2] = delta_z
    A[3] = phi
    return A


# def quadrupole(Mass_center, num, r_1, r_3, delta_x, delta_y, delta_z):
#    # Функция, расчитывающая квадрупольный вклад
#    r_5 = r_3 * r_1 * r_1
#    r_7 = r_5 * r_1 * r_1
#    DR = (Mass_center[num, 7] * delta_x * delta_y
#          + Mass_center[num, 8] * delta_x * delta_z
#          + Mass_center[num, 9] * delta_y * delta_z) * 5
#    a_x = - (Mass_center[num, 7] * delta_y + Mass_center[num, 8] * delta_z) \
#        / r_5 + DR * delta_x / r_7
#    a_y = - (Mass_center[num, 7] * delta_x + Mass_center[num, 9] * delta_z) \
#        / r_5 + DR * delta_y / r_7
#    a_z = - (Mass_center[num, 8] * delta_x + Mass_center[num, 9] * delta_y) \
#        / r_5 + DR * delta_z / r_7
#    phi = DR / (5 * r_5)
#    return np.array([a_x, a_y, a_z, - phi])
