import numpy as np
from eeggan.Bachelorarbeit.Table import Table
import pandas

# --------------------------------------------------------------------------

class GanTable(Table):
    # Class for the latex table for the GAN information

    def __init__(self):
        super().__init__(0, 0)

    # --------------------------------------------------------------------------

    def to_latex(self):
        """
        Turns this table into the latex code and returns it
        """

        self.array = np.array(self.array)
        df = pandas.DataFrame(self.array)
        df.columns = self.x_axis

        latex = df.to_latex(
            bold_rows=True,
            index=False,
            label="tab:test_test",
            caption=self.caption,
        )

        return latex

    # --------------------------------------------------------------------------
