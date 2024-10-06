import os
import re
import pathlib

class EquationsSystemParser:
    def __init__(self):
        self.equations = []
        self.matrix_A = []
        self.vector_B = []
        self.variables_X = []

    def parse(self, file_path: str="input_file.txt"):
        cwd = os.getcwd()
        file_path = os.path.join(cwd, file_path)
        with open(file_path, 'r') as file:
            for line in file:
                equation = line.strip()
                self.equations.append(equation)

    def get_equations(self):
        return self.equations

    def parse_equations(self):
        pattern = r'([+-]?\s*\d*)([a-z])'

        for equation in self.equations:
            equation = equation.replace(" ", "")
            equation = equation.split("=")
            self.vector_B.append(int(equation[1]))

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
                    coeff = int(coeff.strip())

                if var not in self.variables_X:
                    self.variables_X.append(var)

                row_A.append(coeff)

            self.matrix_A.append(row_A)

    def get_matrix_A(self):
        return self.matrix_A

    def get_vector_B(self):
        return self.vector_B

    def get_variables_X(self):
        return self.variables_X

    def get_determinant(self, matrix=None):
        if matrix is None:
            matrix = self.matrix_A

        if len(matrix) == 3:
            a, b, c = matrix[0]
            d, e, f = matrix[1]
            g, h, i = matrix[2]
            return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h)

        elif len(matrix) == 2:
            a, b = matrix[0]
            c, d = matrix[1]
            return a * d - b * c

        return None

    def get_trace(self):
        trace = 0
        for i in range(len(self.variables_X)):
            trace += self.matrix_A[i][i]

        return trace

    def get_vector_norm(self):
        norm = 0
        for i in range(len(self.variables_X)):
            norm += self.vector_B[i] ** 2

        return norm ** 0.5

    def get_transpose(self, matrix=None):
        if matrix is None:
            matrix = self.matrix_A

        transpose = []
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix)):
                row.append(matrix[j][i])
            transpose.append(row)

        return transpose

    def matrix_vector_multiplication(self, matrix, vector):
        result = []
        for i in range(len(matrix)):
            row = matrix[i]
            sum = 0
            for j in range(len(row)):
                sum += row[j] * vector[j]
            result.append(sum)

        return result

    def solve_system_cramer(self):
        determinant = self.get_determinant()
        if determinant == 0:
            return None

        solutions = []
        for i in range(len(self.variables_X)):
            matrix_A_copy = [row[:] for row in self.matrix_A]
            for j in range(len(self.variables_X)):
                matrix_A_copy[j][i] = self.vector_B[j]

            determinant_i = self.get_determinant(matrix=matrix_A_copy)
            solutions.append(determinant_i / determinant)

        return solutions

    def get_cofactor_matrix(self, matrix):
        cofactor_matrix = []
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix)):
                minor_matrix = [row[:] for row in matrix]
                minor_matrix.pop(i)
                for k in range(len(minor_matrix)):
                    minor_matrix[k].pop(j)

                minor_determinant = self.get_determinant(matrix=minor_matrix)
                row.append((-1) ** (i + j) * minor_determinant)

            cofactor_matrix.append(row)

        return cofactor_matrix

    def get_adjugate_matrix(self, matrix):
        cofactor_matrix = self.get_cofactor_matrix(matrix)
        adjugate_matrix = self.get_transpose(cofactor_matrix)
        return adjugate_matrix

    def get_inverse_matrix(self, matrix):
        determinant = self.get_determinant(matrix)
        if determinant == 0:
            return None

        adjugate_matrix = self.get_adjugate_matrix(matrix)
        inverse_matrix = [[element / determinant for element in row] for row in adjugate_matrix]
        return inverse_matrix

    def solve_system_using_inversion(self):
        inverse_matrix = self.get_inverse_matrix(self.matrix_A)
        if inverse_matrix is None:
            return None

        solutions = self.matrix_vector_multiplication(inverse_matrix, self.vector_B)
        return solutions


if __name__ == "__main__":
    parser = EquationsSystemParser()
    parser.parse()
    parser.parse_equations()
    print(f"Matrix A: \n{parser.get_matrix_A()}")
    print(f"Vector B: {parser.get_vector_B()}")
    print(f"Variables: {parser.get_variables_X()}")

    determinant = parser.get_determinant()
    if determinant is not None:
        print(f"Determinant: {determinant}")

    trace = parser.get_trace()
    print(f"Trace: {trace}")

    norm = parser.get_vector_norm()
    print(f"Vector norm: {norm}")

    transpose = parser.get_transpose()
    print(f"Transpose: \n{transpose}")

    result = parser.matrix_vector_multiplication(parser.get_matrix_A(), parser.get_vector_B())
    print(f"A multiplied by B: {result}")

    solutions = parser.solve_system_cramer()
    if solutions is not None:
        print(f"Solution for the system using Cramer Rule: {solutions}")
    else:
        print("The system has no solution or infinite solutions.")

    solutions = parser.solve_system_using_inversion()
    if solutions is not None:
        print(f"Solution for the system using inversion: {solutions}")
    else:
        print("The system has no solution or infinite solutions.")


    #Bonus: the cofactor determinant is equal to the determinant of the original matrix raised to the power of n-1