import numpy as np
from scipy import optimize
from numpy import linalg, array
import znum.Znum as xusun
import helper.Beast as bst

class Math:
    METHOD = 'highs-ipm'
    PRECISION = 6
    operations = {
        '+': lambda x, y: round(x + y, Math.PRECISION),
        '-': lambda x, y: round(x - y, Math.PRECISION),
        '*': lambda x, y: round(x * y, Math.PRECISION),
        '/': lambda x, y: round(x / y, Math.PRECISION),
    }

    def __init__(self, root):
        self.root: xusun.Znum = root

    @staticmethod
    def get_membership(Q, n):
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
        left_part = (Q[1] - Q[0]) / self.root.left
        right_part = (Q[3] - Q[2]) / self.root.right

        Q_int_value = np.concatenate(([Q[0] + i * left_part for i in range(self.root.left + 1)],
                                      [Q[2] + i * right_part for i in range(self.root.right + 1)]))

        Q_int_memb = np.array([self.get_membership(Q, i) for i in Q_int_value])
        return {'value': Q_int_value, 'memb': Q_int_memb}

    def get_matrix(self):
        # # # # # # # #
        # goal_programming = 2
        #
        # self.root.A_int, self.root.B_int = self.get_intermediate(self.root.A), self.get_intermediate(self.root.B)
        # i37, size = self.get_i37(self.root.A_int), len(self.root.A_int['value'])
        # c, bounds = array(np.concatenate(([0] * size, [1] * goal_programming))), [(0, 1)] * (size + goal_programming)
        #
        # A_eq = array([
        #     np.concatenate((self.root.A_int['memb'], [1 if i % 2 == 0 else 1 for i in range(goal_programming)])),
        #     np.concatenate(([1] * size, [0] * goal_programming)),
        #     np.concatenate((self.root.A_int['value'], [0] * goal_programming))
        # ])
        # # # # # # # #

        # # # # # # #
        self.root.A_int, self.root.B_int = self.get_intermediate(self.root.A), self.get_intermediate(self.root.B)
        i37, size = self.get_i37(self.root.A_int), len(self.root.A_int['value'])
        c, bounds = self.root.A_int['memb'], np.full((size, 2), (0, 1), dtype=tuple)
        A_eq = array([self.root.A_int['memb'], np.ones(size), self.root.A_int['value']])
        # # # # # # #


        # # # # # # # # #
        # goal_programming = 6
        #
        # self.root.A_int, self.root.B_int = self.get_intermediate(self.root.A), self.get_intermediate(self.root.B)
        # i37, size = self.get_i37(self.root.A_int), len(self.root.A_int['value'])
        # c, bounds = array(np.concatenate(([0] * size, [1] * goal_programming))), [(0, 1)] * (size + goal_programming)
        #
        # A_eq = array([
        #     np.concatenate((self.root.A_int['memb'], [-1, 1, 0, 0, 0, 0])),
        #     np.concatenate(([1] * size, [0, 0, -1, 1, 0, 0])),
        #     np.concatenate((self.root.A_int['value'], [0, 0, 0, 0, -1, 1]))
        # ])
        # # # # # # # # #



        return tuple(zip(*[
            optimize.linprog(
                c,
                A_eq=A_eq,
                b_eq=array((b20, 1, i37)),
                bounds=bounds,
                method=Math.METHOD).x \
            for b20 in self.root.B_int['value']
        ]))

    @staticmethod
    def get_i37(Q_int):
        return np.dot(Q_int['value'], Q_int['memb']) / np.sum(Q_int['memb'])

    @staticmethod
    def get_Q_from_matrix(matrix):
        Q = np.empty(4)

        Q[0] = min(matrix, key=lambda x: x[0])[0]
        Q[3] = max(matrix, key=lambda x: x[0])[0]

        matrix = list(filter(lambda x: x[1] == 1, matrix))

        Q[1] = min(matrix, key=lambda x: x[0])[0]
        Q[2] = max(matrix, key=lambda x: x[0])[0]

        Q = [round(i, 6) for i in Q]
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
        matrix,  matrix1, matrix2 = [], number_z1.math.get_matrix(), number_z2.math.get_matrix()
        for i, (A1_int_element_value, A1_int_element_memb) in enumerate(zip(number_z1.A_int['value'], number_z1.A_int['memb'])):
            for j, (A2_int_element_value, A2_int_element_memb) in enumerate(zip(number_z2.A_int['value'], number_z2.A_int['memb'])):
                row = [Math.operations[operation](A1_int_element_value, A2_int_element_value), min(A1_int_element_memb, A2_int_element_memb)]
                matrix.append(row + [element1 * element2 for element1 in matrix1[i] for element2 in matrix2[j] ])
        return matrix

    @staticmethod
    def get_minimized_matrix(matrix):
        minimized_matrix = {}
        for row in matrix:
            if row[0] in minimized_matrix:
                # find max of col2
                minimized_matrix[row[0]][0] = max(minimized_matrix[row[0]][0], row[1])

                # add respective probabilities
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

        size1 = len(Number_z1.B_int['memb'])
        size2 = len(Number_z2.B_int['memb'])

        for i, column in enumerate(matrix_by_column):
            row = [sum([i * j for i, j in zip(column1, column)]),
                   min(Number_z1.B_int['memb'][i // size1], Number_z2.B_int['memb'][i % size2])]
            final_matrix.append(row)
        return final_matrix

    @staticmethod
    def z_solver_main(number_z1, number_z2, operation):
        matrix = Math.get_matrix_main(number_z1, number_z2, operation)
        matrix = Math.get_minimized_matrix(matrix)
        A = Math.get_Q_from_matrix(matrix)
        matrix = Math.get_prob_pos(matrix, number_z1, number_z2)
        B = Math.get_Q_from_matrix(matrix)

        return xusun.Znum(A, B)


