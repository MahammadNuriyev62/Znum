import znum.Znum as xusun
from znum.Beast import Beast


class Promethee:

    @staticmethod
    def solver_main(table: list[list], shouldNormalizeWeight=False):
        weights: list[xusun.Znum] = table[0]
        table_main_part: list[list[xusun.Znum]] = table[1:-1]
        criteria_types: list[str] = table[-1]

        table_main_part_transpose = tuple(zip(*table_main_part))
        for column_number, column in enumerate(table_main_part_transpose):
            Beast.normalize(column, criteria_types[column_number])

        preference_table = Promethee.calculate_preference_table(table_main_part)

        if shouldNormalizeWeight:
            Beast.normalize_weight(weights)

        Promethee.weightage(preference_table, weights)
        Promethee.sum_preferences_of_same_category_pair(preference_table)

        vertical_sum = Promethee.vertical_alternative_sum(preference_table)
        horizontal_sum = Promethee.horizontal_alternative_sum(preference_table)

        # horizontal_sum - vertical_sum
        table_to_sort = Beast.subtract_matrix(horizontal_sum, vertical_sum)

        numerated_table_to_sort = Promethee.numerate(table_to_sort)
        sorted_table = Promethee.sort_numerated_single_column_table(numerated_table_to_sort)

        return sorted_table

    @staticmethod
    def calculate_preference_table(table_main_part):
        table_main_part: list[list[xusun.Znum]]

        preference_table = []
        for indexAlternative, alternative in enumerate(table_main_part):
            alternativeRow = []
            for indexOtherAlternative, otherAlternative in enumerate(table_main_part):
                if indexAlternative != indexOtherAlternative:

                    otherAlternativeRow = []
                    for criteria, otherCriteria in zip(alternative, otherAlternative):
                        (d1, do1) = xusun.Znum.Sort.solver_main(criteria, otherCriteria)
                        (d2, do2) = xusun.Znum.Sort.solver_main(otherCriteria, criteria)
                        d = do1 - do2
                        d = d if d > 0 else 0
                        otherAlternativeRow.append(d)
                    alternativeRow.append(otherAlternativeRow)
                else:
                    alternativeRow.append([])

            preference_table.append(alternativeRow)
        return preference_table

    @staticmethod
    def weightage(preference_table, weights):
        for preferenceByCategoriesByAlternatives in preference_table:
            for preferenceByCategories in preferenceByCategoriesByAlternatives:
                for index, (preferenceByCategory, weight) in enumerate(zip(preferenceByCategories, weights)):
                    preferenceByCategories[index] = weight * preferenceByCategory  # order is Znum() * Number()

    @staticmethod
    def sum_preferences_of_same_category_pair(preference_table):
        for preferenceByCategoriesByAlternatives in preference_table:
            for index, preferenceByCategories in enumerate(preferenceByCategoriesByAlternatives):
                preferenceByCategoriesByAlternatives[index] = Beast.accurate_sum(preferenceByCategories)

    @staticmethod
    def vertical_alternative_sum(preference_table):
        return [Beast.accurate_sum(column) for column in zip(*preference_table)]

    @staticmethod
    def horizontal_alternative_sum(preference_table):
        return [Beast.accurate_sum(row) for row in preference_table]

    @staticmethod
    def numerate(single_column_table):
        single_column_table: list[xusun.Znum]
        return list(enumerate(single_column_table, 1))

    @staticmethod
    def sort_numerated_single_column_table(single_column_table):
        single_column_table: list[xusun.Znum]
        sorted_table = tuple(sorted(single_column_table, reverse=True, key=lambda x: x[1]))
        return sorted_table
