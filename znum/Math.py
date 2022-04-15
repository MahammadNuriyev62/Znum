from scipy import optimize
from numpy import linalg, array
import znum.Znum as xusun

class Math:
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

        Q_int_value = [Q[0] + i * left_part for i in range(self.root.left + 1)] + \
                      [Q[2] + i * right_part for i in range(self.root.right + 1)]

        Q_int_memb = [self.get_membership(Q, i) for i in Q_int_value]
        return {'value': Q_int_value, 'memb': Q_int_memb}

    def get_matrix(self):
        self.root.A_int = self.get_intermediate(self.root.A)
        self.root.B_int = self.get_intermediate(self.root.B)
        i37 = self.get_i37(self.root.A_int)
        matrix = [[None for i in range(len(self.root.A_int['value']))
                   ] for i in range(len(self.root.A_int['value']))]

        c = array(self.root.A_int['memb'])
        A_eq = array([self.root.A_int['memb'],
                      [1, 1, 1, 1, 1, 1, 1], self.root.A_int['value']])
        b_eq = array([0, 1, i37])
        bounds = [(0, 1) for i in range(len(self.root.A_int['value']))]
        for i, b20 in enumerate(self.root.B_int['value']):
            # c = array([-i for i in self.root.A_int['value']])
            b_eq[0] = b20
            result = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex').x
            for j, x in enumerate(result):
                matrix[j][i] = x
        return matrix

    @staticmethod
    def get_i37(Q_int):
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
        matrix1 = number_z1.math.get_matrix()
        matrix2 = number_z2.math.get_matrix()
        for i, (A1_int_element_value, A1_int_element_memb) in enumerate(
                zip(number_z1.A_int['value'], number_z1.A_int['memb'])):
            for j, (A2_int_element_value, A2_int_element_memb) in enumerate(
                    zip(number_z2.A_int['value'], number_z2.A_int['memb'])):
                row = [round(eval(f'{A1_int_element_value}{operation}{A2_int_element_value}'), 3),
                       min(A1_int_element_memb, A2_int_element_memb)]
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
        for i, column in enumerate(matrix_by_column):
            row = [sum([i * j for i, j in zip(column1, column)]),
                   min(Number_z1.B_int['memb'][i // 7], Number_z2.B_int['memb'][i % 7])
                   ]
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