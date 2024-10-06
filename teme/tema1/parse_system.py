import re
import pathlib
from typing import List, Tuple, Optional


def load_system(file_path: pathlib.Path = pathlib.Path("input.txt")) -> Tuple[List[List[float]], List[float]]:
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    equations = []
    matrix_A = []
    vector_B = []
    variables_X = []

    with open(file_path, 'r') as file:
        for line in file:
            equation = line.strip()
            equations.append(equation)

    pattern = r'([+-]?\s*\d*)([a-z])'

    for equation in equations:
        equation = equation.replace(" ", "")
        equation = equation.split("=")
        vector_B.append(float(equation[1]))

        row_A = []
        equation = equation[0]
        matches = re.findall(pattern, equation)
        for match in matches:
            coeff, var = match
            if coeff.strip() in ('', '+'):
                coeff = 1
            elif coeff.strip() == '-':
                coeff = -1
            else:
                coeff = float(coeff.strip())

            if var not in variables_X:
                variables_X.append(var)

            row_A.append(coeff)

        matrix_A.append(row_A)

    return matrix_A, vector_B


def determinant(matrix: List[List[float]]) -> float:
    if len(matrix) == 3:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h)

    elif len(matrix) == 2:
        a, b = matrix[0]
        c, d = matrix[1]
        return a * d - b * c

    return 0.0


def trace(matrix: List[List[float]]) -> float:
    trace_sum = 0
    for i in range(len(matrix)):
        trace_sum += matrix[i][i]

    return trace_sum


def norm(vector: List[float]) -> float:
    norm_value = 0
    for i in range(len(vector)):
        norm_value += vector[i] ** 2

    return norm_value ** 0.5


def transpose(matrix: List[List[float]]):
    transpose_matrix = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transpose_matrix.append(row)

    return transpose_matrix


def multiply(matrix: List[List[float]], vector: List[float]) -> List[float]:
    result = []
    for i in range(len(matrix)):
        row = matrix[i]
        summ = 0
        for j in range(len(row)):
            summ += row[j] * vector[j]
        result.append(summ)

    return result

def solve_cramer(matrix: List[List[float]], vector: List[float]) -> Optional[List[float]]:
    determinant_value = determinant(matrix=matrix)
    if determinant_value == 0:
        return None

    solutions = []
    for i in range(len(matrix)):
        matrix_A_copy = [row[:] for row in matrix]
        for j in range(len(matrix)):
            matrix_A_copy[j][i] = vector[j]

        determinant_i = determinant(matrix=matrix_A_copy)
        solutions.append(determinant_i / determinant_value)

    return solutions

def cofactor(matrix: List[List[float]]) -> List[List[float]]:
    cofactor_matrix = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            minor_matrix = [row[:] for row in matrix]
            minor_matrix.pop(i)
            for k in range(len(minor_matrix)):
                minor_matrix[k].pop(j)

            minor_determinant = determinant(matrix=minor_matrix)
            row.append((-1) ** (i + j) * minor_determinant)

        cofactor_matrix.append(row)

    return cofactor_matrix

def adjoint(matrix: List[List[float]]) -> List[List[float]]:
    cofactor_matrix = cofactor(matrix=matrix)
    adjugate_matrix = transpose(matrix=cofactor_matrix)
    return adjugate_matrix

def get_inverse_matrix(matrix: List[List[float]]) -> Optional[List[List[float]]]:
    determinant_value = determinant(matrix=matrix)
    if determinant_value == 0:
        return None

    adjugate_matrix = adjoint(matrix=matrix)
    inverse_matrix = [[element / determinant_value for element in row] for row in adjugate_matrix]
    return inverse_matrix

def solve(matrix: List[List[float]], vector: List[float]) -> Optional[List[float]]:
    inverse_matrix = get_inverse_matrix(matrix=matrix)
    if inverse_matrix is None:
        return None

    solutions = multiply(matrix=inverse_matrix, vector=vector)
    return solutions


if __name__ == "__main__":
    A, B = load_system(pathlib.Path("input.txt"))

    print(f"Matrix A: \n{A}")
    print(f"Vector B: {B}")

    print(f"Determinant: {determinant(matrix=A)}")

    print(f"Trace: {trace(matrix=A)}")

    print(f"Vector norm: {norm(vector=B)}")

    print(f"Transpose: \n{transpose(matrix=A)}")

    print(f"A multiplied by B: {multiply(matrix=A, vector=B)}")

    cramer_solutions = solve_cramer(matrix=A, vector=B)
    if cramer_solutions is not None:
        print(f"Solution for the system using Cramer Rule: {cramer_solutions}")
    else:
        print("The system has no solution or infinite solutions.")

    inversion_solutions = solve(matrix=A, vector=B)
    if inversion_solutions is not None:
        print(f"Solution for the system using inversion: {inversion_solutions}")
    else:
        print("The system has no solution or infinite solutions.")


    #Bonus: the cofactor determinant is equal to the determinant of the original matrix raised to the power of n-1