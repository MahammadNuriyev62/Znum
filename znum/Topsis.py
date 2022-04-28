from pprint import pprint

import znum.Znum as xusun
from znum.Beast import Beast


class Topsis:

    class DataType:
        ALTERNATIVE = "A"
        CRITERIA = "C"
        TYPE = "TYPE"

    @staticmethod
    def solver_main(table: list[list], shouldNormalizeWeight=False):
        """
        table[0] -> weights
        table[1:-1] -> main part
        table[-1] -> criteria types
        :param shouldNormalizeWeight:
        :param table:
        :return:
        """
        weights: list[xusun.Znum] = table[0]
        table_main_part: list[list[xusun.Znum]] = table[1:-1]
        criteria_types: list[str] = table[-1]
        main_table_part_transpose = tuple(zip(*table_main_part))
        for column_number, column in enumerate(main_table_part_transpose):
            Beast.normalize(column, criteria_types[column_number])

        if shouldNormalizeWeight:
            Topsis.normalize_weight(weights)


        Topsis.weightage(table_main_part, weights)

        table_1 = Topsis.get_table_n(table_main_part, 1)
        table_0 = Topsis.get_table_n(table_main_part, 0)

        s_best = Topsis.find_extremum(table_1)
        s_worst = Topsis.find_extremum(table_0)
        p = Topsis.find_distance(s_best, s_worst)

        return p



    @staticmethod
    def normalize_weight(weights: list):
        weights: list[xusun.Znum]
        znum_sum = weights[0]
        for weight in weights[1:]:
            znum_sum += weight
        for i, znum in enumerate(weights):
            weights[i] = znum / znum_sum

    @staticmethod
    def weightage(table_main_part, weights):
        for row in table_main_part:
            for i, (znum, weight) in enumerate(zip(row, weights)):
                row[i] = znum * weight

    @staticmethod
    def get_table_n(table_main_part, n: int):
        table_main_part: list[list[xusun.Znum]]
        table_n = []
        for row in table_main_part:
            row_n = []
            for znum in row:
                number = sum([abs(n - p) for p in znum.A + znum.B]) * 0.5
                row_n.append(number)
            table_n.append(row_n)
        return table_n

    @staticmethod
    def find_extremum(table_n: list[list[int]]):
        return [sum(row) for row in table_n]

    @staticmethod
    def find_distance(s_best, s_worst):
        return [worst / (best + worst) for best, worst in zip(s_best, s_worst)]
