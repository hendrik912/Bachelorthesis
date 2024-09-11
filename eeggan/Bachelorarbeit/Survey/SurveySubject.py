import numpy as np

class SurveySubject:
    # This class stores all the information of a person who participated in the survey
    # such as the country of residence, the profession etc., as well as the ratings the
    # person provided.

    def __init__(self, id):
        self.id = id
        self.country = ""
        self.profession = ""
        self.experience = ""
        self.notes = ""
        self.role = ""
        self.weight = 1
        self.conf_ratings_all = []
        self.conf_ratings_real = []
        self.conf_ratings_fake = []
        self.noise_ratings_all = []
        self.noise_ratings_real = []
        self.noise_ratings_fake = []

    def __str__(self):
        """
        Turns the object into a string that contains all the information associated with that person.
        """
        notes = ""
        for c in self.notes:
             if c.isprintable():
                notes += c

        output = "subject " + str(self.id)
        output += "\n\tCountry of residence: " + self.country
        output += "\n\tProfession: " + self.profession
        output += "\n\tRole: " + self.role
        output += "\n\tExperience: " + self.experience
        output += "\n\tNotes: " + notes

        output += "\n\tAvg conf real: " + str(np.average(self.conf_ratings_real))
        output += "\n\tAvg conf fake: " + str(np.average(self.conf_ratings_fake))

        output += "\n\tAvg noise real: " + str(np.average(self.noise_ratings_real))
        output += "\n\tAvg noise fake: " + str(np.average(self.noise_ratings_fake))

        return output

# --------------------------------------------------------------------------