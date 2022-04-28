import znum.Znum as xusun



class Beast:
    class CriteriaType:
        COST = "C"
        BENEFIT = "B"


    @staticmethod
    def sum(array):
        array: list[xusun.Znum]

        result = xusun.Znum([0, 0, 0, 0], [1, 1, 1, 1])
        for znum in array:
            if type(znum) is xusun.Znum:
                result += znum
        return result

    @staticmethod
    def accurate_sum(array):
        array: list[xusun.Znum]

        array = list(filter(lambda x: x, array))
        if len(array) == 0: return None

        result = array[0]
        for znum in array[1:]:
            if type(znum) is xusun.Znum:
                result += znum
        return result

    @staticmethod
    def subtract_matrix(o1, o2):
        return [z1 - z2 for z1, z2 in zip(o1, o2)]


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