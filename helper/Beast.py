from pprint import pprint

from znum.Znum import Znum


class Beast:
    ZNUM_SIZE = 8

    class Methods:
        TOPSIS = 1
        PROMETHEE = 2

    @staticmethod
    def read_xlsx_main():
        filename = Beast.get_file_path()
        return Beast.read_xlsx(filename)

    @staticmethod
    def read_znums_from_xlsx(method=None):
        method = method or Beast.Methods.TOPSIS
        table = Beast.read_xlsx_main()
        if method == Beast.Methods.TOPSIS:
            return Beast.parse_znums_from_table_for_topsis(table)
        elif method == Beast.Methods.PROMETHEE:
            return Beast.parse_znums_from_table_for_promethee(table)
        else:
            raise Exception('Invalid Optimization Method for input Table')


    @staticmethod
    def get_file_path():
        import tkinter.filedialog
        filename = tkinter.filedialog.askopenfile()
        return filename.name

    @staticmethod
    def read_xlsx(filename: str):
        from openpyxl import load_workbook, Workbook
        workbook: Workbook = load_workbook(filename)

        table = []
        for sheet in workbook:
            for row in sheet.rows:
                values = [o.value for o in row]
                any(values) and table.append(values)

        return table

    @staticmethod
    def parse_znums_from_table_for_topsis(table: list[list]):
        weights, extra, main, types = table[0], table[1], table[2: -1], table[-1]

        weights_modified = Beast.parse_znums_from_row(weights[1:])
        main_modified = [Beast.parse_znums_from_row(row[1:]) for row in main]
        types_modified = [t for t in types[1:] if t]
        return [weights_modified, *main_modified, types_modified]

    @staticmethod
    def parse_znums_from_table_for_promethee(table: list[list]):
        weights, extra, main = table[0], table[1], table[2:]

        weights_modified = Beast.parse_znums_from_row(weights[1:])
        main_modified = [Beast.parse_znums_from_row(row[1:]) for row in main]
        return [weights_modified, *main_modified]

    @staticmethod
    def parse_znums_from_row(row: list[int]):
        row_modified = []
        for i in range(0, len(row), Beast.ZNUM_SIZE):
            znumAsList = row[i:i + Beast.ZNUM_SIZE]
            index = int(Beast.ZNUM_SIZE / 2)
            znum = Znum(A=znumAsList[:index], B=znumAsList[index:])
            row_modified.append(znum)
        return row_modified

    @staticmethod
    def save_znums_as_one_column_grouped_by_criteria(table):
        from openpyxl import Workbook

        table: list[list[Znum]]

        workbook = Workbook()
        sheet = workbook.create_sheet("XUSUN")
        table_transpose = zip(*table)
        for row in table_transpose:
            for znum in row:
                sheet.append(znum.A + znum.B)
        workbook.save('output.xlsx')
