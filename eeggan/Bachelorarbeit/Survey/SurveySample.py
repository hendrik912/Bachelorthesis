
class SurveySample:
    # This class represents a question in the survey and stores the ratings it was given by the participants

    def __init__(self, id):
        self.id = id
        self.real = None
        self.conf_real = []
        self.conf_real_weighted = []
        self.noise = []
        self.noise_weighted = []

    def __str__(self):
        """
        Turn object into string
        """

        output = ""
        output += "id: " + str(self.id)
        output += "\nreal: " + str(self.real)
        output += "\nconf_real" + str(self.conf_real)
        output += "\nnoise" + str(self.noise) + "\n"
        return output

# --------------------------------------------------------------------------
