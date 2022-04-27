import znum.Znum as xusun



class Beast:

    @staticmethod
    def sum(array):
        array: list[xusun.Znum]

        result = xusun.Znum([0, 0, 0, 0], [1, 1, 1, 1])
        for znum in array:
            if type(znum) is xusun.Znum:
                result += znum
        return result

    @staticmethod
    def subtract_matrix(o1, o2):
        return [z1 - z2 for z1, z2 in zip(o1, o2)]