import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import geopy.distance
from equation_solvers import partial_pivot_gauss, gauss_seidel, jacobi, gauss_seidel_sparse, sparse_iterative, sparse_LU
from equation_solvers import gauss_seidel_effective


def extract_data_from_txt(file_path, nb_points=-1):
    data = pd.read_csv(file_path, delimiter=",").values
    x1, x2, y = data[:nb_points, 1], data[:nb_points, 0], data[:nb_points-1, 2]
    n = len(x1)
    x = np.zeros(n-1)
    for i in range(n-1):
        x[i] = geopy.distance.distance((x1[i], x2[i]), (x1[i+1], x2[i+1])).km
        x[i] += x[i-1]

    return x, y


def spline_interpolation(x, y, mode="numpy", optimized_gauss_seidel=True):
    if mode == "numpy":
        linear_solve_eq = np.linalg.solve
    elif mode == "PG":
        linear_solve_eq = partial_pivot_gauss
    elif mode == "IT":
        linear_solve_eq = jacobi
    elif mode == "GS":
        linear_solve_eq = gauss_seidel
    elif mode == "IS":
        linear_solve_eq = gauss_seidel_sparse
    elif mode == "ScipyLU":
        linear_solve_eq = sparse_LU
    elif mode == "ScipyIterative":
        linear_solve_eq = sparse_iterative

    n = len(x)
    M = np.zeros(len(x), np.float64)
    A = np.zeros((len(x), len(x)), dtype=np.float64)
    A[0, 0] = A[-1, -1] = 1
    for i in range(0, n-2):
        hi = x[i+1] - x[i]
        hi_2 = x[i+2] - x[i+1]
        A[i+1][i:i+3] = [hi, 2*(hi+hi_2), hi_2]

    for i in range(1, n-1):
        M[i] = 3 * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - 3 * (y[i] - y[i - 1]) / (x[i] - x[i - 1])
    M = np.transpose(M)
    a = np.zeros(n-1)
    b = np.zeros(n-1)
    if mode == "IS" and optimized_gauss_seidel:
        c = linear_solve_eq(A, M, optimized=True)
    else:
        c = linear_solve_eq(A, M)
    d = np.zeros(n-1)

    for i in range(n-1):
        a[i] = y[i]
        b[i] = ((y[i + 1] - y[i]) / (x[i + 1] - x[i])) - ((x[i + 1] - x[i]) / 3) * (2 * c[i] + c[i + 1])
        d[i] = (c[i + 1] - c[i]) / (3 * (x[i + 1] - x[i]))

    x_ = []
    y_ = []
    for i in range(n-1):
        for p in np.linspace(x[i], x[i + 1], 100):
            x_.append(p)
            y_.append(a[i] + b[i] * (p - x[i]) + c[i] * (p - x[i]) ** 2 + d[i] * (p - x[i]) ** 3)

    return x_, y_


def extract_test_data(x, y, take_every_nth=3):
    x_test = x[::take_every_nth]
    y_test = y[::take_every_nth]
    x_interpolate = [xi for i, xi in enumerate(x) if i % 3 != 0]
    y_interpolate = [yi for i, yi in enumerate(y) if i % 3 != 0]

    return np.array(x_interpolate), np.array(y_interpolate), x_test, y_test


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def calc_interpolation_err(x, y, x_test, y_test):
    err = np.zeros(len(x_test))
    for i in range(len(x_test)):
        idx = find_nearest(x, x_test[i])
        err[i] = np.abs(y_test[i] - y[idx])

    avg_err = np.mean(err)
    print(f"Average error of interpolation: {avg_err}")
    return avg_err


def main():
    data_path = "data/trasa2.txt"
    nb_points = -1
    x, y = extract_data_from_txt(data_path, nb_points)
    x, y, x_test, y_test = extract_test_data(x, y)

    x_spline_pg, y_spline_pg = spline_interpolation(x, y, mode="PG")
    calc_interpolation_err(x_spline_pg, y_spline_pg, x_test, y_test)

    x_spline_gs, y_spline_gs = spline_interpolation(x, y, mode="GS")
    calc_interpolation_err(x_spline_gs, y_spline_gs, x_test, y_test)

    x_spline_it, y_spline_it = spline_interpolation(x, y, mode="IT")
    calc_interpolation_err(x_spline_it, y_spline_it, x_test, y_test)

    x_spline_is, y_spline_is = spline_interpolation(x, y, mode="IS", optimized_gauss_seidel=False)
    calc_interpolation_err(x_spline_is, y_spline_is, x_test, y_test)
    
    x_spline_is_opt, y_spline_is_opt = spline_interpolation(x, y, mode="IS", optimized_gauss_seidel=True)
    calc_interpolation_err(x_spline_is_opt, y_spline_is_opt, x_test, y_test)

    x_spline_lu, y_spline_lu = spline_interpolation(x, y, mode="ScipyLU")
    calc_interpolation_err(x_spline_lu, y_spline_lu, x_test, y_test)

    x_spline_sparse_it, y_spline_sparse_it = spline_interpolation(x, y, mode="ScipyIterative")
    calc_interpolation_err(x_spline_sparse_it, y_spline_sparse_it, x_test, y_test)

    plt.plot(x, y, 'o', label='Real values')
    plt.plot(x_test, y_test, 'x', label="Test values")
    plt.plot(x_spline_pg, y_spline_pg, label='PG')
    plt.plot(x_spline_gs, y_spline_gs, label='GS')
    plt.plot(x_spline_it, y_spline_it, label='IT')
    plt.plot(x_spline_is, y_spline_is, label='IS')
    plt.plot(x_spline_is_opt, y_spline_is_opt, label='IS optimized')
    plt.plot(x_spline_lu, y_spline_lu, label="Sparse LU")
    plt.plot(x_spline_sparse_it, y_spline_sparse_it, label="Sparse Iterative")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

