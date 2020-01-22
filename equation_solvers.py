import numpy as np
from timeit import default_timer as timer
import scipy.sparse.linalg


def partial_pivot_gauss(a, b, eps=1e-6):
    start_partial_pivoting = timer()
    n = len(b)
    A = np.copy(a)
    b = np.copy(b)
    X = np.zeros(n)
    for k in range(0, n-1):
        if np.abs(A[k, k]) < eps:
            for i in range(k+1, n):
                if np.abs(A[i, k]) > np.abs(A[k, k]):
                    A[[k, i]] = A[[i, k]]
                    b[[k, i]] = b[[i, k]]
                    break
        # Gauss elimination
        for i in range(k+1, n):
            if A[k, k] == 0:
                continue
            coeff = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] = A[i, j] - coeff*A[k, j]
            b[i] = b[i] - coeff*b[k]
    
    # Reverse Gauss procedure
    X[n-1] = b[n-1]/A[n-1, n-1]
    for i in reversed(range(0, n-1)):
       sum_ax = 0
       for j in range(i+1, n):
           sum_ax += A[i, j] * X[j]
       X[i] = (b[i] - sum_ax)/A[i, i]
    
    stop_partial_pivoting = timer()
    print(f"(PG) Partial pivoting gauss equation solve time: {stop_partial_pivoting - start_partial_pivoting}")
    
    return X


def gauss_seidel_update_x(A, x, b):
    n = len(x)
    for i in range(n):
        temp = b[i]
        for j in range(n):
            if i != j:
                temp -= A[i][j] * x[j]
        x[i] = temp / A[i][i]
        
    return x


def gauss_seidel_update_x_sparse(compact_matrix, x, b):
    n = len(x)
    for i in range(n):
        temp = b[i]
        for j in range(compact_matrix.shape[1]):
            row_index = int(compact_matrix[0, j])
            column_index = int(compact_matrix[1, j])
            value = compact_matrix[2, j]
            if row_index == i:
                if row_index == column_index:
                    diag_el = value
                else:
                    temp -= value * x[column_index]
        
        x[i] = temp / diag_el
    # print(f"x: {x}")
        
    return x

def gauss_seidel_update_x_sparse_optimized(compact_matrix, x, b):
    n = len(x)
    m = compact_matrix.shape[1]
    temp = np.zeros(n)
    for j in range(m):
        row_index = int(compact_matrix[0, j])
        column_index = int(compact_matrix[1, j])
        value = compact_matrix[2, j]
        if row_index == column_index:
            temp[row_index] += b[row_index]
            diag_el = value
        else:
            temp[row_index] -= value * x[column_index]
            
        if (j == m-1) or (int(compact_matrix[0, j+1]) > row_index):
            x[row_index] = temp[row_index]/diag_el        
    return x

def gauss_seidel(A, b, x=None, stop_criterion=1e-6, nb_iteration=20, check_criterion=False):
    test_convergence_gauss_seidel(A)
    start_gauss_seidel = timer()
    if x is None:
        x = np.zeros(len(b), dtype=np.float32)

    for i in range(nb_iteration):
        x_old = np.copy(x) 
        gauss_seidel_update_x(A, x, b)
        if check_criterion:
            if x.all() != 0:
                update_score = np.abs((x - x_old)/x)
                if (update_score <= stop_criterion).all():
                    break
        
    stop_gauss_seidel = timer()
    print(f"(GS) Gauss-Seidel number of iteration: {i+1} to converge in {stop_criterion} update, time consumption: {stop_gauss_seidel - start_gauss_seidel}")
    
    return x


def gauss_seidel_effective(A, b, x=None, n=20):
    if x is None:
        x = np.zeros(len(b), dtype=np.float32)
    start_gs = timer()
    L = np.tril(A)
    U = A - L
    for i in range(n):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))

    stop_gs = timer()
    print(f"GS: Time consumption: {stop_gs - start_gs}")
    return x


def gauss_seidel_sparse(A_sparse, b, x=None, nb_iteration=20, stop_criterion=1e-6, check_criterion=False, optimized=False):
    test_convergence_gauss_seidel(A_sparse)
    compact_matrix = get_compact_matrix(A_sparse)
    start_gauss_seidel = timer()
    if x is None:
        x = np.zeros(len(b), dtype=np.float32)
        
    for i in range(nb_iteration):
        x_old = np.copy(x)
        if optimized:
            gauss_seidel_update_x_sparse_optimized(compact_matrix, x, b)
        else: 
            gauss_seidel_update_x_sparse(compact_matrix, x, b)
        if check_criterion:
            if x.all() != 0:
                update_score = np.abs((x - x_old)/x)
                if (update_score <= stop_criterion).all():
                    break
        
    stop_gauss_seidel = timer()
    print(f"(IS) Gauss-Seidel Sparse: optimized: {optimized} number of iteration: {i+1} to converge in {stop_criterion} update, time consumption: {stop_gauss_seidel - start_gauss_seidel}")
    
    return x


def jacobi(A, b, nb_iter=25, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    start_jacobi = timer()
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = np.zeros(len(b), dtype=np.float32)

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = np.diag(A)
    R = A - np.diagflat(D)
    # Iterate for N times                                                                                                                                                                          
    for i in range(nb_iter):
        x = (b - np.dot(R,x)) / D

    stop_jacobi = timer()
    print(f"(IT) Jacobi: number of iteration {i+1} time consumption: {stop_jacobi-start_jacobi}")
    return x


def get_compact_matrix(sparse_matrix):
    non_zero_elements_idxs = np.argwhere(sparse_matrix != 0)
    row_indexes = non_zero_elements_idxs[:, 0]
    column_indexes = non_zero_elements_idxs[:, 1]
    nb_elements = len(non_zero_elements_idxs)
    
    compact_matrix = np.zeros((3, nb_elements), dtype=np.float32)
    compact_matrix[0, :] = row_indexes
    compact_matrix[1, :] = column_indexes
    compact_matrix[2, :] = sparse_matrix[row_indexes, column_indexes]
    
    return compact_matrix


def sparse_LU(sparse_matrix, b):
    start_LU = timer()
    A = scipy.sparse.csc_matrix(sparse_matrix)
    LU = scipy.sparse.linalg.splu(A)
    x = LU.solve(b)
    stop_LU = timer()
    print(f"Sparse LU scipy solver time consumption: {stop_LU - start_LU}")
    return x


def sparse_iterative(sparse_matrix, b, criterion=1e-5, max_iter=None):
    start_iterative = timer()
    x = scipy.sparse.linalg.bicg(sparse_matrix, b, tol=criterion, maxiter=max_iter)[0]
    stop_iterative = timer()
    print(f"Sparse iterative method using scipy solver time consumption: {stop_iterative - start_iterative}")
    return x


def test_convergence_gauss_seidel(A):
    converges = True
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            sum_j = 0
            if i != j:
                sum_j += np.abs(A[i, j])
                if np.abs(A[i, i]) < sum_j:
                    converges = False        
    if converges:
        print("Gauss-seidel: The convergence test has been passed")
    else:
        print("\nGauss-seidel iteration method to solve linear equation is not converging!\n")
                
    
def test_partial_pivoting(a, b, numpy_solution):        
    solution = partial_pivot_gauss(a, b)
    print(f"Partial pivot solution : {solution}")
    avg_err = np.mean(np.abs(numpy_solution - solution))
    print(f"Average error between numpy and partial pivoting: {avg_err}")


def test_gauss_seidel(a, b, numpy_solution, stop_criterion=1e-6, nb_iter=20):
    solution = gauss_seidel(a, b, stop_criterion=stop_criterion, nb_iteration=nb_iter)
    print(f"Gauss-Seidel solution : {solution}")
    avg_err = np.mean(np.abs(numpy_solution - solution))
    print(f"Average error between numpy and Gauss-Seidel: {avg_err}")


def test_compact_matrix(sparse_matrix):
    compact_matrix = get_compact_matrix(sparse_matrix)
    print("Sparse matrix")
    print(sparse_matrix)
    print("Compact matrix")
    print(compact_matrix)


def test_sparse_GS(a, b, numpy_solution, stop_criterion=1e-6, nb_iter=20, optimized=True):
    solution = gauss_seidel_sparse(a, b, stop_criterion=stop_criterion, nb_iteration=nb_iter, optimized=optimized)
    print(f"Sparse Gauss-Seidel solution : {solution}")
    avg_err = np.mean(np.abs(numpy_solution - solution))
    print(f"Average error between numpy and Sparse Gauss-Seidel: {avg_err}")


def test_sparse_scipy(a, b, numpy_solution):
    x_lu = sparse_LU(a, b)
    avg_err = np.mean(np.abs(numpy_solution - x_lu))
    print(f"Average error between numpy and Sparse LU: {avg_err}")
    x_iterative = sparse_iterative(a, b)
    avg_err = np.mean(np.abs(numpy_solution - x_iterative))
    print(f"Average error between numpy and iterative sparse scipy: {avg_err}")

    
if __name__ == "__main__":
    # a = np.array([[0, 31, -1, 3, 1, -15],
    #             [0, 32, 0, -11, 70, -3],
    #             [61, 2, 22, -12, -1, 22],
    #             [-2, 17, 24, 0, 2, -6],
    #             [3, 0, 14, -27, 1, -5],
    #             [62, 31, -4, 5, 2, 0]], dtype=np.float32)
    # b = np.array([55, 47, 22, 3, 4, 8], dtype=np.float32)
    
    a = np.array([[4, 1, 2],[3, 5, 1],[1, 1, 3]], dtype=np.float32) 
    b = np.array([4,7,3], dtype=np.float32) 
    # a_sparse = np.array([[0, 1, 0],[3, 2, 0],[0, 3, 2]], dtype=np.float32) 
    # a = a_sparse
    # test_compact_matrix(a_sparse)
    numpy_solution = np.linalg.solve(a, b)
    print(f"Numpy linalg solution: {numpy_solution}")
    test_partial_pivoting(a, b, numpy_solution)
    test_gauss_seidel(a, b, numpy_solution, nb_iter=25)
    test_sparse_GS(a, b, numpy_solution, nb_iter=25, stop_criterion=1e-6, optimized=False)
    test_sparse_GS(a, b, numpy_solution, nb_iter=25, stop_criterion=1e-6)
    test_sparse_scipy(a, b, numpy_solution)
