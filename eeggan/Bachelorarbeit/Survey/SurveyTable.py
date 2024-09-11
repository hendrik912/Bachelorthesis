import numpy as np
from eeggan.Bachelorarbeit.Table import Table
import pandas

# --------------------------------------------------------------------------

class SurveyTable(Table):
    # Class for the latex tables for the survey

    def __init__(self, table_type):
        super().__init__(0, 0)
        self.table_type = table_type

    # --------------------------------------------------------------------------

    def to_string(self):
        """
        turns table into string
        """

        output = ""
        for row in self.array:
            output += str(row) + "\n"
        return output

    # --------------------------------------------------------------------------

    def to_latex(self):
        """
        turns the table into the latex code
        """

        self.array = np.array(self.array)
        df = pandas.DataFrame(self.array)

        df.columns = self.x_axis
        latex = df.to_latex(
            bold_rows=True,
            index=False,
            caption=self.caption,
        )

        return latex

    # --------------------------------------------------------------------------
