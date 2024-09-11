import os
import mne
import warnings
from eeggan.Bachelorarbeit import persistence
import eeggan.Bachelorarbeit.GAN.evaluate as g_evaluate
import eeggan.Bachelorarbeit.GAN.train as g_train
import eeggan.Bachelorarbeit.Classifier.evaluate as c_evaluate
import eeggan.Bachelorarbeit.Classifier.train as c_train
import eeggan.Bachelorarbeit.Survey.create as s_create
import eeggan.Bachelorarbeit.Survey.evaluate as s_evaluate
import eeggan.Bachelorarbeit.constants as constants
import eeggan.Bachelorarbeit.config as config
import eeggan.Bachelorarbeit.GAN.metrics as metrics

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# --------------------------------------------------------------------------

# The path to the data
# This folder is one level above the project folder
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), "Data", "ieaexperiment/")

# The path where the results will be stored (models and plats)
# This folder is one level above the project folder
RESULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), "BA_Data/Results/")

print(RESULT_PATH)
#RESULT_PATH = "/share/temp/students/heilts/results/"

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# --------------------------------------------------------------------------

if __name__ == '__main__':
    """
    Here all the necessary steps from the creation of the datasets, to the training of the models and 
    the evaluation of the results are called. Which subset of steps is done is determined by the config-file. 
    There you can also change some parameters, that are needed for the particular step. 

    Steps:
        - create datasets
        - train GANs
        - train base classifiers 
            (not augmented, they are used for filtering the generated samples by the classification confidence)
        - create survey
        - train classifiers on augmented datasets
        - evaluate GANs
        - evaluate classifiers
        - evaluate the survey
        
    The config file is located in the base directory
    """

    # --------------------------------------------------------------------------

    # the test mode reduces the number of epochs and the sizes of the datasets etc.
    # in order to test the functionality of the pipeline
    if config.test_mode:
        # GAN Training
        config.gan_epochs = 15
        config.gan_bs = 256

        # Classifier Training
        config.c_use_clfr_th = True
        config.c_use_jsd_threshold = True
        config.c_epochs = 1
        config.c_pred_threshold = 0.0
        config.c_jsd_threshold = 9999999.00
        config.c_iterations = 2
        config.c_ratio_start = 0
        config.c_ratio = 2

        # Create Survey
        config.s_jsd_threshold = 9999999.00
        config.s_pred_threshold = 0.0

        # GAN Eval
        config.eval_times = 2
        config.eval_num_samples = 15

        # Clfr Eval

    # --------------------------------------------------------------------------

    sample_lengths = [1, 2, 3, 4, 6]

    # Create new datasets of the given window lengths
    if config.create_datasets_b:
        print("> create datasets")
        sample_lengths = [1, 2, 3, 4, 6]
        persistence.create_datasets(DATA_PATH, RESULT_PATH, sample_lengths, config.data_filter,
                                    filter_range=(config.data_hpass, config.data_lpass))

    # --------------------------------------------------------------------------

    # Train the GANs
    if config.train_gan_b:
        # Each row in the list below is for a GAN with specific parameters:
        # sl: The sample length or window length in seconds
        # bs: batch size, i.e. the amount of samples taken in a training loop
        # fade: whether to use the fading-technique or not
        # c0: train class 0
        # c1: train class 1, c0 and c1 only make sense if 'comb' is false
        # comb: stands for 'combined', i.e. the GAN is trained for both classes
        # The other parameters are determined by the config file

        parameters = [
            # sl, bs, fade,  c0,  c1,  comb
            (6, 128, True, True, True, False),
            (4, 128, True, True, True, False),
            (3, 128, True, True, True, False),
            (2, 128, True, True, True, False),
            (1, 128, True, True, True, False),

            (6, 128, True, True, True, True),
            (4, 128, True, True, True, True),
            (3, 128, True, True, True, True),
            (2, 128, True, True, True, True),
            (1, 128, True, True, True, True),
        ]
        g_train.train_gan(parameters, RESULT_PATH, config.gan_learning_rate, config.gan_epochs,
                          constants.FS)

    # --------------------------------------------------------------------------

    # Train the classifiers for datasets without augmentation
    if config.train_base_classifiers_b:
        # For each entry in the sample_lengths-list a classifier will be trained
        # the other parameters are determined by the config-file

        sample_lengths = [1,2,3,4,6]
        c_train.train_base_classifiers(sample_lengths, RESULT_PATH, config.c_batch_size, config.c_iterations,
                                       config.c_epochs)

    # --------------------------------------------------------------------------

    # Traing the classifiers for the augmented datasets
    if config.train_augm_classifiers_b:
        # For each entry in the parameter list a classifier will be trained.
        # The parameters determine which GAN will be used to augment the dataset.
        # the other parameters are determined by the config-file

        parameters = [
            # sl, bs, fade, comb
            (6, 128, True, False),
            (4, 128, True, False),
            (3, 128, True, False),
            #(2, 128, True, False),
            #(1, 128, True, False),
        ]

        data_divisions = [8,1]

        for p in parameters:
            sl = p[0]
            bs = p[1]
            fade = p[2]
            comb = p[3]
            print("SL, BS, Fade, Comb")
            print(sl, bs, fade, comb, '\n')

            # for each value in data_divisions a set of classifiers will be trained of the dataset divided by div,
            # so only len(dataset)/div samples are used.
            for div in data_divisions:
                output = c_train.train_augm_classifiers(div, sl, bs, comb, RESULT_PATH, fade, config.gan_stage,
                                                        config.c_ratio_start, config.c_ratio, config.c_filter_range,
                                                        config.c_batch_size, config.c_epochs, config.c_use_clfr_th,
                                                        config.c_pred_threshold, config.c_use_jsd_threshold,
                                                        config.c_jsd_threshold, config.c_iterations, metrics.jsd)
                print("div:", div)
                print(output)

    # --------------------------------------------------------------------------

    # Create the time and frequency plots for the survey
    if config.create_survey_b:
        s_create.create_survey(RESULT_PATH, config.s_sample_len, config.s_bs, config.s_fading,
                               config.s_separate_gans, config.s_num_imgs, constants.CHNS, constants.FS,
                               config.s_jsd_threshold, config.s_pred_threshold)

    # --------------------------------------------------------------------------

    # This part is for evaluation the trained GANs
    if config.eval_gans_b:
        print("> Evaluating GANs")

        # The entries in the list below are used to identify the GANs which are to be evaluated and if the data should
        # be postprocessed before the evaluation.
        # sl: sample lengths in seconds
        # bs: batch size
        # fade: whether to use the fading-technique or not
        # comb: true if its a model which was trained on both classes
        # pp: true if the data should be postprocessed before the evaluation
        parameters = [
            # sl, bs, fade, comb, pp
            #(1, 128, True, True, True),
            #(1, 128, True, True, False),
            #(1, 128, True, False, True),
            (1, 128, True, False, False),
            #(2, 128, True, True, True),
            #(2, 128, True, True, False),
            #(2, 128, True, False, True),
            (2, 128, True, False, False),
            #(3, 128, True, True, True),
            #(3, 128, True, True, False),
            #(3, 128, True, False, True),
            (3, 128, True, False, False),
            #(4, 128, True, True, True),
            #(4, 128, True, True, False),
            #(4, 128, True, False, True),
            (4, 128, True, False, False),
            #(6, 128, True, True, True),
            #(6, 128, True, True, False),
            #(6, 128, True, False, True),
            (6, 128, True, False, False),
        ]

        # get the threshold for the classification confidence
        c_pred_threshold = config.c_pred_threshold

        # get the threshold for the filtering by the jensen shannon distance
        c_jsd_threshold = config.c_jsd_threshold

        print("jsd metric", c_jsd_threshold)
        g_evaluate.evaluate(RESULT_PATH, config.eval_num_samples, parameters, config.eval_times,
                            threshold=c_jsd_threshold, pred_th=c_pred_threshold, calc_metrics=config.eval_g_calc_metrics_b,
                            metric=metrics.jsd)

    # --------------------------------------------------------------------------

    # evaluate the classifiers that were trained on the augmented datasets
    if config.eval_classifiers_b:
        print("> Evaluating Classifiers")

        # The entries in the list below are used to identify the classifiers which are to be evaluated
        # sl: Sample length in seconds
        # bs: batch size
        # fade: whether to use the fading-technique or not
        # the other parameters are determined by the config-file
        parameters = [
            # sl, bs, fade
            #(1, 128, True),
            #(2, 128, True),
            (3, 128, True),
            (4, 128, True),
            (6, 128, True),
        ]

        c_evaluate.evaluate(parameters, config.c_ratio, config.c_iterations, RESULT_PATH,
                            calc_metrics=config.eval_c_calc_metrics_b)

    # --------------------------------------------------------------------------

    # This section evaluates the survey and produces plots as well as tables of the data from the survey
    if config.eval_survey_b:
        print("> Evaluating Survey")

        path = os.path.join(RESULT_PATH, "survey")
        # the name of the file with the results of the survey
        result_fn = "eeggan_survey.csv"

        record_fn = "all_subjects_4s_bs128_fade_combined_jsd20_scale3_BEST/record.csv"
        # to use the record created with a different gan, copy the record.csv file into survey-folder
        #record_fn = "record.csv"
        s_evaluate.evaluate(path, result_fn, record_fn, BA_path = RESULT_PATH)

    # -------------------------------------------------------------------