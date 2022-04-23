import znum.Znum as xusun


class Beast:

    class CriteriaType:
        COST = "C"
        BENEFIT = "B"

    class DataType:
        ALTERNATIVE = "A"
        CRITERIA = "C"
        TYPE = "TYPE"

    def __init__(self, root):
        self.root: xusun.Znum = root

    @staticmethod
    def solver_main(table: tuple[(str, (str, tuple))], shouldNormalizeWeight=True):
        """
        table[0] -> weights
        table[1:-1] -> main part
        table[-1] -> criteria types
        :param shouldNormalizeWeight:
        :param table:
        :return:
        """
        table = list(table)
        main_table_part = table[1:-1]
        criteria_types = table[-1]
        main_table_part_transpose = tuple(zip(*main_table_part))
        for column_number, column in enumerate(main_table_part_transpose):
            Beast.normalize(column, criteria_types[column_number])

        if shouldNormalizeWeight:
            table[0] = Beast.normalize_weight(table[0])

        Beast.weightage(table)

        table_1 = Beast.get_table_n(table, 1)
        table_0 = Beast.get_table_n(table, 0)


        s_best = Beast.find_extremum(table_1)
        s_worst = Beast.find_extremum(table_0)
        p = Beast.find_distance(s_best, s_worst)

        return p


    @staticmethod
    def normalize(znums_of_criteria: list, criteria_type: str):
        criteria_type_mapper = {
            Beast.CriteriaType.COST: Beast.normalize_cost,
            Beast.CriteriaType.BENEFIT: Beast.normalize_benefit,
        }
        criteria_type_mapper.get(criteria_type, criteria_type_mapper[Beast.CriteriaType.COST])(znums_of_criteria)

    @staticmethod
    def normalize_benefit(znums_of_criteria: list):
        all_a = [a for znum in znums_of_criteria for a in znum.A]
        max_a = max(all_a)
        for znum in znums_of_criteria:
            znum.A = [a / max_a for a in znum.A]

    @staticmethod
    def normalize_cost(znums_of_criteria: list):
        all_a = [a for znum in znums_of_criteria for a in znum.A]
        min_a = min(all_a)
        for znum in znums_of_criteria:
            znum.A = list(reversed([min_a / a for a in znum.A]))

    @staticmethod
    def normalize_weight(weights: list):
        weights: list[xusun.Znum]
        znum_sum = weights[0]
        for weight in weights[1:]:
            znum_sum += weight
        return tuple(znum / znum_sum for znum in weights)

    @staticmethod
    def weightage(table):
        for index in range(1, len(table) - 1):
            table[index] = [znum * weight for znum, weight in zip(table[index], table[0])]

    @staticmethod
    def get_table_n(table, n: int):
        table: list[list[xusun.Znum]]
        table_n = []
        for row in table[1:-1]:
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
