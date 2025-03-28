import numpy as np
from qiskit.quantum_info import SparsePauliOp, pauli_basis
import sympy as sp


def create_random_hermitian_matrix(n, seed=None):
    """
    Create a Hermitian matrix of size n x n.

    Args:
        n (int): The size of the matrix (n x n).

    Returns:
        observable (np.ndarray): A random Hermitian observable.
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    # Ensure Hermitian property
    B = (A + A.conj().T) / 2
    return B


def create_parametrized_hermitian_matrix(n):
    """
    Create a parametrized Hermitian matrix of size n x n.

    Args:
        n (int): The size of the matrix (n x n).

    Returns:
        observable (np.ndarray): A parametrized Hermitian matrix.
    """
    B_sym = sp.Matrix(np.array(
        [sp.symbols(f'x_{i}_{j}') for i in range(n) for j in range(n)]).reshape(n, n))

    # Ensure Hermitian property
    B_sym = (B_sym + B_sym.conjugate().T) / 2
    return B_sym


def assign_parameters_to_matrix(matrix, parameters):
    """
    Assign parameters to a parametrized matrix.

    Args:
        matrix (np.ndarray): A parametrized matrix.
        parameters (np.ndarray): The values to assign to the parameters.

    Returns:
        matrix (np.ndarray): The matrix with the parameters assigned.
    """
    # Extract symbols in a sorted order (only needed once)
    symbols = sorted(matrix.free_symbols, key=lambda x: str(x))

    # Create a fast numerical function for evaluation
    f_matrix = sp.lambdify(symbols, matrix, 'numpy')

    # Convert to numpy array and cast to complex type
    return np.array(f_matrix(*parameters.ravel()), dtype=complex)


def hermitian_to_sparsepauliop(B, n_qubits):
    """Convert a Hermitian matrix B to a SparsePauliOp representation."""
    basis = pauli_basis(n_qubits)

    # Compute the coefficients for each Pauli string
    coefficients = [np.trace(B @ P.to_matrix()) / (2**n_qubits) for P in basis]

    return SparsePauliOp(basis, coefficients)


def random_complex_vector(n, seed=None, range_real=[0, 1], range_imag=[0, 1]):
    """
    Create a random complex vector of size n.

    Args:
        n (int): The size of the vector.
        seed (int): Optional. The seed for the random number generator.
        range_real (list): Optional. The range for the real part of the vector elements.
        range_imag (list): Optional. The range for the imaginary part of the vector elements

    Returns:
        vector (np.ndarray): A random complex vector.
    """
    if seed is not None:
        np.random.seed(seed)
    real = np.random.uniform(range_real[0], range_real[1], n)
    imag = np.random.uniform(range_imag[0], range_imag[1], n)
    return real + 1j * imag
