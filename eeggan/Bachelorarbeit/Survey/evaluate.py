import os
import numpy as np
import eeggan.Bachelorarbeit.util as util
import eeggan.Bachelorarbeit.Survey.plot as s_plot
import eeggan.Bachelorarbeit.persistence as persistence
from eeggan.Bachelorarbeit.Survey.SurveySubject import SurveySubject
from eeggan.Bachelorarbeit.Survey.SurveySample import SurveySample
from eeggan.Bachelorarbeit.Survey.TableType import TableType
from eeggan.Bachelorarbeit.Survey.SurveyTable import SurveyTable
import matplotlib.pyplot as plt
import scipy
from scipy.stats import ks_2samp, chisquare, fisher_exact, mannwhitneyu
from decimal import Decimal

from decimal import Decimal
# codes for real or fake samples in survey:
REAL_INDICES = [1, 2, 4, 5, 9]
FAKE_INDICES = [0, 3, 6, 7, 8]

# Confidence:  MannwhitneyuResult(statistic=4061.5, pvalue=0.9742929619325416)
# Noise:       MannwhitneyuResult(statistic=5341.0, pvalue=0.00018770889846827292)

# weighted:

# MannwhitneyuResult(statistic=4937.0, pvalue=0.8759097308988408)
# MannwhitneyuResult(statistic=6510.0, pvalue=0.00018831387835342492)


def evaluate(path, survey_result_fn, survey_record_fn, BA_path):
    """
    This function handles the evaluation of the survey and creates tables as well as plots

    Parameters
    ----------
    path: Path to results
    survey_result_fn: filename of the survey results
    survey_record_fn: the filename of the file that contains information about the survey
                      (i.e. which samples are real and which are fake)
    """

    BA_path_imgs = os.path.join(BA_path, "eval", "survey", "images")
    BA_path_tbls = os.path.join(BA_path, "eval", "survey", "tables")

    subjects = prepare_survey_results_for_subjects(path, survey_result_fn, survey_record_fn)
    samples = prepare_survey_results_for_samples(path, survey_result_fn, survey_record_fn)

    sim_vals = []
    real_vals = []

    for s in samples:
        if s.real:
            real_vals += s.conf_real
        else:
            sim_vals += s.conf_real

    x_vals = [i for i in range(1, 11)]

    y_sim = [0 for _ in range(10)]
    y_real = [0 for _ in range(10)]

    for i in range(len(sim_vals)):
        y_sim[sim_vals[i]-1] += 1
        y_real[real_vals[i]-1] += 1

    print(scipy.stats.mannwhitneyu(sim_vals, real_vals))

    # --------

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.bar(x_vals, y_real, alpha=0.5, label="real")
    ax1.bar(x_vals, y_sim, alpha=0.5, label="sim")
    ax1.set_xticks(x_vals)
    ax1.title.set_text("a)")
    ax1.legend()

    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Confidence')

    #plt.savefig("survey_dist_conf.png")
    #plt.show()


    # ----------------------------------------------

    sim_noise = []
    real_noise = []

    for s in samples:
        if s.real:
            real_noise += s.noise
        else:
            sim_noise += s.noise

    x_vals = [i for i in range(1, 11)]
    y_sim = [0 for _ in range(10)]
    y_real = [0 for _ in range(10)]

    for i in range(len(sim_vals)):
        y_sim[sim_noise[i]-1] += 1
        y_real[real_noise[i]-1] += 1

    print(scipy.stats.mannwhitneyu(sim_noise, real_noise))

    # --------

    ax2.bar(x_vals, y_real, alpha=0.5, label="real")
    ax2.bar(x_vals, y_sim, alpha=0.5, label="sim")
    ax2.set_xticks(x_vals)
    ax2.legend()

    # ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Perceived Noise')
    ax2.title.set_text("b)")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    w = 8 * 0.75
    h = 4 * 0.75

    fig.set_size_inches(w, h)

    plt.savefig("survey_dist.pdf", bbox_inches='tight')
    plt.show()

    """
    # -----------------------------------------------------------------------

    # print(ks_2samp(sim_vals, real_vals))
    # print(fisher_exact(sim_vals, real_vals, ddof=0, axis=0))

    if False:
        output = s_plot.get_subjects_as_string(subjects)
        persistence.string_to_file(output, BA_path_tbls, "subject_wise.txt")
        table_types = [TableType.IMAGE_WISE, TableType.SUBJECT_WISE, TableType.ACROSS_ALL_SUBJ]

        for tt in table_types:
            if tt == TableType.IMAGE_WISE:
                data = samples
            else:
                data = subjects

            table = ratings_to_table(data, tt)

            print(table.to_latex())

            persistence.string_to_file(table.to_latex(), BA_path_tbls, "latex_survey_table_" + tt.value + ".tex")

        s_plot.plot_ratings(subjects, BA_path_imgs, show=False, show_titles=True)
    
    """
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------

def ratings_to_table(data, tableType):
    """
    Takes the ratings (data), puts them into the table given by TableType and returns the table.
    """

    if tableType == TableType.SUBJECT_WISE:
        """
        Subm. | conf all | conf real | conf fake | noise all | noise real | noise fake |
        1     |      1.0 |       1.0 |       1.0 |       1.0 |        1.0 |        1.0 |
        """
        table = SurveyTable(tableType)
        header = ["Subm.", "avg. conf. real", "avg. conf. fake",
                  "avg. noise real", "avg. noise fake"]

        table.set_x_axis(header)
        table.array = []
        table.title = tableType.value
        table.caption = "Darstellung der Ergebnisse der Umfrage pro Antwort."

        for s in data:
            row = []
            row.append(str(int(s.id)+1))
            row.append(format(np.average(s.conf_ratings_real),".2f"))
            row.append(format(np.average(s.conf_ratings_fake),".2f"))
            row.append(format(np.average(s.noise_ratings_real),".2f"))
            row.append(format(np.average(s.noise_ratings_fake),".2f"))
            table.array.append(row)
        return table
    elif tableType == tableType.IMAGE_WISE:
        """
        real/fake | Img_id | conf_avg | noise_avg | conf_range | noise_range |
        """
        table = SurveyTable(tableType)
        header = ["Data", "id", "conf. avg.", "weigh. c.a.", "noise avg.", "weigh. n.a.", "conf. range", "noise range"]
        table.set_x_axis(header)
        table.array = []
        table.title = tableType.value
        table.caption = "Darstellung der Ergebnisse der Umfrage pro Frage."

        real_subset = [s for s in data if s.real]
        fake_subset = [s for s in data if not s.real]

        for j in range(0, 2):
            use_real = j==0

            if j==1:
                table.array.append([" " for _ in range(len(header))])

            for i in range(0, len(real_subset)):
                row = []
                if i == 0:
                    if use_real: row.append("Real")
                    else: row.append("Fake")
                else:
                    row.append(" ")
                s = real_subset[i] if use_real else fake_subset[i]

                row.append(str(int(s.id) + 1))

                row.append(format(np.average(s.conf_real), ".2f"))
                row.append(format(np.average(s.conf_real_weighted), ".2f"))

                row.append(format(np.average(s.noise),".2f"))
                row.append(format(np.average(s.noise_weighted), ".2f"))

                row.append([np.min(s.conf_real), np.max(s.conf_real)])
                row.append([np.min(s.noise), np.max(s.noise)])
                table.array.append(row)
        return table


    elif tableType == tableType.ACROSS_ALL_SUBJ:
        """
        Modality | conf all | conf real | conf fake | noise all | noise real | noise fake |
        avg      |
        med      |
        range    |
        """

        header = [" ", "Modalitaet", "total conf.", "conf. real", "conf. fake",
                  "total noise", "noise real", "noise fake"]
        table = SurveyTable(tableType)
        table.set_x_axis(header)
        table.array = []
        table.title = tableType.value
        table.caption = "Darstellung der Ergebnisse der Umfrage."

        for i in range(0, 2):
            if i==1:
                row = [" " for _ in header]
                table.array.append(row)

            weighted = i==1

            ratings = []
            ratings.append(get_rating_for_type(data, weighted, "conf_all"))
            ratings.append(get_rating_for_type(data, weighted, "conf_real"))
            ratings.append(get_rating_for_type(data, weighted, "conf_fake"))
            ratings.append(get_rating_for_type(data, weighted, "noise_all"))
            ratings.append(get_rating_for_type(data, weighted, "noise_real"))
            ratings.append(get_rating_for_type(data, weighted, "noise_fake"))

            weighted_str = "ungew." if i==0 else "gew."
            row = [weighted_str, "avg"]
            for r in ratings:
                row.append(format(np.average(r),".2f"))
            table.array.append(row)

            row = [" ", "median"]
            for r in ratings:
                row.append(np.median(r))
            table.array.append(row)

            row = [" ", "range"]
            for r in ratings:
                row.append([np.min(r), np.max(r)])
            table.array.append(row)

        # print(table)
        return table
    else:
        return None

# --------------------------------------------------------------------------

def get_rating_for_type(subjects, weighted, type):
    """
    iterate over subjects and return the ratings for the given type (e.g. noise of fake samples)
    and applies the given weight.
    """

    ratings = []

    for s in subjects:
        weight = s.weight if weighted else 1
        for i in range(weight):
            if type == "conf_all":
                ratings.append(s.conf_ratings_all)
            elif type == "conf_real":
                ratings.append(s.conf_ratings_real)
            elif type == "conf_fake":
                ratings.append(s.conf_ratings_fake)
            elif type == "noise_all":
                ratings.append(s.noise_ratings_all)
            elif type == "noise_real":
                ratings.append(s.noise_ratings_real)
            else:
                ratings.append(s.noise_ratings_fake)

    return ratings

# --------------------------------------------------------------------------

def prepare_survey_results_for_subjects(path, survey_result_fn, survey_record_fn):
    """
    Parses the survey results into the SurveySubject-Objects

    Parameters
    ----------
    path: path where the survey results are located
    survey_result_fn: the fileeame of the survey results
    survey_record_fn: the filename of the file that contains information about the survey
    (i.e. which samples are real and which are fake)

    Returns
    -------
    list of SurveySubject
    """


    res_cols, res_vals = persistence.read_csv(os.path.join(path, survey_result_fn))
    _, ref_vals = persistence.read_csv(os.path.join(path, survey_record_fn))

    real = "real"
    subjects = []
    prefixes = [str(i) + "." + str(j) for i in range(1, 21) for j in range(1, 3)]
    other = {"country": "Country", "profession": "What is your professional background?",
             "role": "What is your role?", "exp": "How", "notes": "No"}

    for k in range(0, len(res_vals)):
        subj = SurveySubject(k)
        subjects.append(subj)
        weight = 1

        for i in range(0, len(res_cols)):
            rc = res_cols[i]
            b, j = util.get_index_of_prefixes(rc, prefixes)

            if b:
                prefix = prefixes[j]
                ref_index = int(prefix[0:(prefix.find("."))]) - 1
                ref_val = ref_vals[0][ref_index]

                if prefix.endswith(".1"):
                    subj.conf_ratings_all.append(res_vals[k][i])
                    if ref_val == real:
                        subj.conf_ratings_real.append(res_vals[k][i])
                    else:
                        subj.conf_ratings_fake.append(res_vals[k][i])
                else:
                    subj.noise_ratings_all.append(res_vals[k][i])
                    if ref_val == real:
                        subj.noise_ratings_real.append(res_vals[k][i])
                    else:
                        subj.noise_ratings_fake.append(res_vals[k][i])
            else:
                if isinstance(res_vals[k][i], str):
                    res = res_vals[k][i]
                else:
                    res = ""

                if rc.startswith(other["profession"]):
                    subjects[-1].profession = res
                elif rc.startswith(other["role"]):
                    subjects[-1].role = res
                elif rc.startswith(other["notes"]):
                    subjects[-1].notes = res
                elif rc.startswith(other["country"]):
                    subjects[-1].country = res
                elif rc.startswith(other["exp"]):
                    subjects[-1].experience = res
                    years_exp = res_vals[k][i]
                    if years_exp == "less than 5 years":
                        weight = 1
                    elif years_exp == "between 5 and 10 years":
                        weight = 2
                    elif years_exp == "more than 10 years":
                        weight = 3
                    subjects[-1].weight = weight

    return subjects

# --------------------------------------------------------------------------

def prepare_survey_results_for_samples(path, survey_result_fn, survey_record_fn):
    """
    Parses the survey results into the SurveySample-Objects

    Parameters
    ----------
    path: path where the survey results are located
    survey_result_fn: the fileeame of the survey results
    survey_record_fn: the filename of the file that contains information about the survey
    (i.e. which samples are real and which are fake)

    Returns
    -------
    list of SurveySamples
    """

    res_cols, res_vals = persistence.read_csv(os.path.join(path, survey_result_fn))
    _, ref_vals = persistence.read_csv(os.path.join(path, survey_record_fn))

    prefixes = [str(i) + "." + str(j) for i in range(1, 21) for j in range(1, 3)]

    samples = []
    for i in range(0, 20):
        samples.append(SurveySample(i+1))

    subjects = prepare_survey_results_for_subjects(path, survey_result_fn, survey_record_fn)

    for i in range(0, len(res_vals)):
        weight = subjects[i].weight

        for j in range(0, len(res_cols)):
            question = res_cols[j]

            if question.startswith(tuple(prefixes)):
                point_loc = question.find(".")
                index = int(question[0:(question.find("."))]) - 1
                ref_val = ref_vals[0][index]

                samples[index].real = (ref_val == "real")

                if question[point_loc:].startswith(".1"):
                    samples[index].conf_real.append(res_vals[i][j])
                    for _ in range(0,weight):
                        samples[index].conf_real_weighted.append(res_vals[i][j])
                else:
                    samples[index].noise.append(res_vals[i][j])
                    for _ in range(0, weight):
                        samples[index].noise_weighted.append(res_vals[i][j])
    return samples

# --------------------------------------------------------------------------
