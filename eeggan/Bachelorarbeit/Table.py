import numpy as np

class Table:
    # The base class for the latex tables for the evaluation of GANs, classifiers and the survey

    def __init__(self, num_rows, num_cols):
        self.array = np.array([[None for _ in range(0, num_cols)] for _ in range(0, num_rows)])
        self.caption = None
        self.x_axis = []
        self.y_axis = []
        self.x_axis_label = None
        self.y_axis_label = None
        self.title = ""

    # --------------------------------------------------------------------------

    def __str__(self):
        """
        Returns
        -------
        Table as a string
        """
        output = "Caption: " + self.caption
        output += "\nx_axis: " + self.x_axis_label + " " + str(self.x_axis)
        output += "\ny_axis: " + self.y_axis_label + " " + str(self.y_axis)

        for row in self.array:
            line = ""
            for col in row:
                line += str(col) + ", "
            output += "\n" + line[:-2]
        return output + "\n"

    # --------------------------------------------------------------------------

    def to_latex(self):
        """

        Returns
        -------
        Table as latex (should to be overwritten)
        """
        return ""

    # --------------------------------------------------------------------------

    def _title_to_filename(self):
        """
        Turns the title of the table into a filename

        Returns
        -------
        filename as str
        """

        fn = self.title.replace(' ', '_')
        fn = fn.replace(',', '')
        fn = fn.replace(':', '')
        fn = fn.replace('/', '_')
        fn = fn.replace('.', '')
        return fn + ".tex"

    # --------------------------------------------------------------------------

    def get_title(self):
        """
        Returns
        -------
        The title
        """
        return self._title_to_filename()

    # --------------------------------------------------------------------------

    def set_x_axis(self, array):
        """
        Sets the x axis

        Parameters
        ----------
        array

        """
        self.x_axis = [str(a) for a in array]

    # --------------------------------------------------------------------------

    def set_y_axis(self, array):
        """
        Sets the y axis

        Parameters
        ----------
        array

        """
        self.y_axis = [str(a) for a in array]

    # --------------------------------------------------------------------------
