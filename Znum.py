from scipy import optimize
from numpy import linalg, array
# from Replacer import replacer_reverse, round_sig

class Znum:
    def __init__(self, A, B, left=3, right=2, precision=3):
        self.A = A
        self.B = B
        self.precision = precision
        self.left, self.right = left, right
        self.A_int = None  # self.get_intermediate(A)
        self.B_int = None  # self.get_intermediate(B)

    def checker(self, Q):
        return True if Q[0] <= Q[1] <= Q[2] <= Q[3] else False

    def checker_B(self):
        return all([0<=b<=1 for b in self.B])

    def get_membership(self, Q, n):
        if Q[0] < n < Q[1]:
            # kx + b = y
            x1, y1, x2, y2 = Q[0], 0, Q[1], 1
            i, j = array([[x1, 1], [x2, 1]]), array([y1, y2])
            k, b = tuple(linalg.solve(i, j))
            return k * n + b
        elif Q[2] < n < Q[3]:
            x1, y1, x2, y2 = Q[2], 1, Q[3], 0
            i, j = array([[x1, 1], [x2, 1]]), array([y1, y2])
            k, b = tuple(linalg.solve(i, j))
            return k * n + b
        elif Q[1] <= n <= Q[2]:
            return 1
        else:
            return 0

    def get_intermediate(self, Q):
        left_part = (Q[1] - Q[0]) / self.left
        right_part = (Q[3] - Q[2]) / self.right

        Q_int_value = [Q[0] + i * left_part for i in range(self.left + 1)] + \
                      [Q[2] + i * right_part for i in range(self.right + 1)]

        Q_int_memb = [self.get_membership(Q, i) for i in Q_int_value]
        return {'value': Q_int_value, 'memb': Q_int_memb}

    def get_matrix(self):
        self.A_int = self.get_intermediate(self.A)
        self.B_int = self.get_intermediate(self.B)
        i37 = self.get_i37(self.A_int)
        matrix = [[None for i in range(len(self.A_int['value']))
                   ] for i in range(len(self.A_int['value']))]

        c = array(self.A_int['memb'])
        A_eq = array([self.A_int['memb'],
                      [1, 1, 1, 1, 1, 1, 1], self.A_int['value']])
        b_eq = array([0, 1, i37])
        bounds = [(0, 1) for i in range(len(self.A_int['value']))]
        for i, b20 in enumerate(self.B_int['value']):
            # c = array([-i for i in self.A_int['value']])
            b_eq[0] = b20
            result = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex').x
            for j, x in enumerate(result):
                matrix[j][i] = x
        return matrix

    def get_i37(self, Q_int):
        i37 = 0
        for i, j in zip(Q_int['value'], Q_int['memb']):
            i37 += i * j
        return i37 / sum(Q_int['memb'])

    @staticmethod
    def get_Q_from_matrix(matrix):
        Q = [None, None, None, None]

        Q[0] = min(matrix, key=lambda x: x[0])[0]
        Q[3] = max(matrix, key=lambda x: x[0])[0]

        matrix = list(filter(lambda x: x[1] == 1, matrix))

        Q[1] = min(matrix, key=lambda x: x[0])[0]
        Q[2] = max(matrix, key=lambda x: x[0])[0]

        Q = [round(i, 3) for i in Q]
        return Q

    @staticmethod
    def get_matrix_main(number_z1, number_z2, operation):
        """
        option
        1 - add,
        2 - sub,
        3 - mul,
        4 - div
        """
        matrix = []
        matrix1 = number_z1.get_matrix()
        matrix2 = number_z2.get_matrix()
        for i, (A1_int_element_value, A1_int_element_memb) in enumerate(
                zip(number_z1.A_int['value'], number_z1.A_int['memb'])):
            for j, (A2_int_element_value, A2_int_element_memb) in enumerate(
                    zip(number_z2.A_int['value'], number_z2.A_int['memb'])):
                row = []
                row.append(round(eval(f'{A1_int_element_value}{operation}{A2_int_element_value}'), 3))
                row.append(min(A1_int_element_memb, A2_int_element_memb))
                for element1 in matrix1[i]:
                    for element2 in matrix2[j]:
                        row.append(element1 * element2)
                matrix.append(row)

        return matrix

    @staticmethod
    def get_minimized_matrix(matrix):
        minimized_matrix = {}
        for row in matrix:
            if row[0] in minimized_matrix:
                # find max of col2
                minimized_matrix[row[0]][0] = max(minimized_matrix[row[0]][0], row[1])

                # add respective propabilities
                for i, n in enumerate(row[2:]):
                    minimized_matrix[row[0]][i + 1] += n
            else:
                minimized_matrix[row[0]] = row[1:]
        minimized_matrix = [[key] + minimized_matrix[key] for key in minimized_matrix]

        return minimized_matrix

    @staticmethod
    def get_prob_pos(matrix, Number_z1, Number_z2):
        matrix_by_column = list(zip(*matrix))
        column1 = matrix_by_column[1]
        matrix_by_column = matrix_by_column[2:]

        final_matrix = []
        for i, column in enumerate(matrix_by_column):
            row = [sum([i * j for i, j in zip(column1, column)]),
                   min(Number_z1.B_int['memb'][i // 7], Number_z2.B_int['memb'][i % 7])
                   ]
            final_matrix.append(row)

        return final_matrix

    @staticmethod
    def z_solver_main(number_z1, number_z2, operation):
        matrix = Znum.get_matrix_main(number_z1, number_z2, operation)
        matrix = Znum.get_minimized_matrix(matrix)
        A = Znum.get_Q_from_matrix(matrix)
        matrix = Znum.get_prob_pos(matrix, number_z1, number_z2)
        B = Znum.get_Q_from_matrix(matrix)

        return Znum(A, B)

    def __str__(self):
        return f"Znum(A={self.A}, B={self.B})"

    def __repr__(self) -> str:
        return f"Znum(A={self.A}, B={self.B})"

    def __add__(self, other):
        return Znum.z_solver_main(self, other, '+')

    def __mul__(self, other):
        return Znum.z_solver_main(self, other, '*')

    def __sub__(self, other):
        return Znum.z_solver_main(self, other, '-')

    def __truediv__(self, other):
        return Znum.z_solver_main(self, other, '/')

    def __pow__(self, power, modulo=None):
        return Znum(A=[a**power for a in self.A], B=self.B.copy())