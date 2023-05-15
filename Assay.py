
import re
import numpy as np
import xlrd


class Assay:
    """
    Programmatic access to ClarioStar plate reader data
    
    author: Harold Fellerman

    Takes an excel file from CLARIOStarand gives read access
    to the contained data:
    The Assay instance will have the following attributes:
    assay.time,  a numpy array of recorded times
    assay.well,  an array with fluorescence measurements for each well
    assay.matix, a 2d array with fluorescence measurements for each well

    Wells are adressed by a single integer index ranging from 0 to 95, e.g. 
    assay.well[14] gives the timeline for well 14 (colum B row 2 in a 12x8 layout).
    """

    def __init__(self, p_path, p_cols, p_rows):
        def parse_time(string):
            """parses the Cycle time string into seconds
            
            Two regular expresion matches to account for the different Cycle time string formats
            """

            # matching for hrs, min, seconds format
            match = re.match(r'Cycle (\d+) \((\d+)(?: h)?(?: (\d+) min)?(?: (\d+) s)?\)', string)
            if match:
                hours = int(match.group(2)) if match.group(2) else 0
                minutes = int(match.group(3)) if match.group(3) else 0
                seconds = int(match.group(4)) if match.group(4) else 0
                total_seconds = hours*3600 + minutes*60 + seconds
                return total_seconds
            
            # matching for min, seconds format
            match = re.match(r'Cycle (\d+) \((\d+) min(?: (\d+) s)?\)', string)
            if match:
                minutes = int(match.group(2))
                seconds = int(match.group(3)) if match.group(3) else 0
                total_seconds = minutes*60 + seconds
                return total_seconds
            print("Failed to read assay data, please check format.")


        self.COLS = p_cols
        self.ROWS = p_rows
        self.path = p_path
        book = xlrd.open_workbook(self.path)
        # open sheets by name
        ptcl = book.sheet_by_name('Protocol Information')
        data = book.sheet_by_name('DATA')
        
        # read protocol
        self.test_name = ptcl.cell(3, 0).value[len('Test Name: '):]
        self.measurement = ptcl.cell(11, 1).value
        self.cycles = int(ptcl.cell(17, 1).value)
        self.cycle_time = float(ptcl.cell(18, 1).value/60)

        # initialize data attributes
        self.time = np.zeros(self.cycles)
        self.well = np.zeros((self.ROWS*self.COLS, self.cycles))
        self.matrix = np.zeros((self.ROWS, self.COLS, self.cycles))


        for idx in range(self.cycles):
            # parse time
            self.time[idx] = parse_time(data.cell(12 + (self.ROWS+4)*idx, 0).value)

            # read wells
            for col in range(self.COLS):
                for row in range(self.ROWS):
                    val = data.cell(15 + (self.ROWS+4)*idx + row,
                                    1 + col).value
                    self.well[row*self.COLS + col, idx] = val if val != '' else None
        # reshape assay.wells into 2d array for easier access
        self.matrix = self.well.reshape(self.ROWS, self.COLS, self.cycles)

    @property
    def path(self):
        return self._path

    @property
    def COLS(self):
        return self._COLS
    
    @property
    def ROWS(self):
        return self._ROWS
    
    @path.setter
    def path(self, value):
        if not isinstance(value, str):
            raise ValueError("path must be of type string")
        else:
            self._path = value

    @COLS.setter
    def COLS(self, value):
        if not isinstance(value, int):
            raise ValueError("cols must be of type int")
        else:
            self._COLS = value
    
    @ROWS.setter
    def ROWS(self, value):
        if not isinstance(value, int):
            raise ValueError("rows must be of type int")
        else:
            self._ROWS = value
