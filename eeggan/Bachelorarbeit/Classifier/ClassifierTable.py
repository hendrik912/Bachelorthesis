import numpy as np
from eeggan.Bachelorarbeit.Table import Table
from eeggan.Bachelorarbeit.Classifier.TableType import TableType
import pandas

# --------------------------------------------------------------------------

class ClassifierTable(Table):
    # Class for the latex table for the GAN information

    def __init__(self, num_rows, num_cols, table_type):
        super().__init__(num_rows, num_cols)
        self.table_type = table_type

    # --------------------------------------------------------------------------

    def to_latex(self):
        """
        Turn the classifier table into latex code and return it
        """

        if self.table_type == TableType.BY_SAMPLE_LEN:
            x_axis = [str(val) for val in self.x_axis]
            section_label = "W.L."
        elif self.table_type == TableType.BY_DIVISION:
            x_axis = [str(val) + "s" for val in self.x_axis]
            section_label = "Div"
        else:
            x_axis = [str(val) for val in self.x_axis]
            section_label = "Ratio"

        self.array = np.array(self.array)

        df = pandas.DataFrame(self.array)

        header = [
            [" "] + [self.x_axis_label] + x_axis,
            [section_label] + [self.y_axis_label] + [" " for _ in x_axis]
        ]

        df.columns = header
        latex = df.to_latex(
            bold_rows=True,
            index=False,
            label="tab:" + section_label,
            caption=self.caption,
        )

        return latex

    # --------------------------------------------------------------------------
