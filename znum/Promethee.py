import znum.Znum as xusun

from pprint import pprint

from znum.Topsis import Topsis
from znum.Beast import Beast

class Promethee:

    @staticmethod
    def solver_main(table: list[list], shouldNormalizeWeight=False):
        weights: list[xusun.Znum] = table[0]
        table_main_part: list[list[xusun.Znum]] = table[1:]

        table_main_part_transpose = tuple(zip(*table_main_part))
        for column_number, column in enumerate(table_main_part_transpose):
            Topsis.normalize(column, Topsis.CriteriaType.BENEFIT)

        preference_table = Promethee.calculate_preference_table(table_main_part)
        Promethee.weightage(preference_table, weights)
        # pprint(preference_table)
        Promethee.sum_preferences_of_same_category_pair(preference_table)
        pprint(preference_table)

        vertical_sum = Promethee.vertical_alternative_sum(preference_table)
        horizontal_sum = Promethee.horizontal_alternative_sum(preference_table)

        # horizontal_sum - vertical_sum
        table_to_sort = Beast.subtract_matrix(horizontal_sum, vertical_sum)
        sorted_table = tuple(sorted(table_to_sort, reverse=True))

        print(sorted_table)

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
        # for row in table_main_part:
        #     for i, (znum, weight) in enumerate(zip(row, weights)):
        #         row[i] = znum * weight

    @staticmethod
    def sum_preferences_of_same_category_pair(preference_table):
        for preferenceByCategoriesByAlternatives in preference_table:
            for index, preferenceByCategories in enumerate(preferenceByCategoriesByAlternatives):
                preferenceByCategoriesByAlternatives[index] = xusun.Znum.Beast.sum(preferenceByCategories)

    @staticmethod
    def vertical_alternative_sum(preference_table):
        return [Beast.sum(column) for column in zip(*preference_table)]

    @staticmethod
    def horizontal_alternative_sum(preference_table):
        return [Beast.sum(row) for row in preference_table]