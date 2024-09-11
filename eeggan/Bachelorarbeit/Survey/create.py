import os
import random
import eeggan.Bachelorarbeit.util as util
import eeggan.Bachelorarbeit.persistence as persistence
import eeggan.Bachelorarbeit.GAN.generate as generate
import eeggan.Bachelorarbeit.GAN.plot as g_plot
import eeggan.Bachelorarbeit.GAN.util as g_util
import eeggan.Bachelorarbeit.GAN.metrics as metrics
from PIL import Image

# --------------------------------------------------------------------------

# codes for real or fake samples in survey
# (the filenames contain one of these numbers in case there is some confusion later on due to shuffling)
REAL_INDICES = [1, 2, 4, 5, 9]
FAKE_INDICES = [0, 3, 6, 7, 8]

# --------------------------------------------------------------------------


def create_survey(result_path, sample_len, bs, fade, seperate_gans, num_imgs, channels, fs, jsd_th, clfr_th):
    """
    Creates the eeg samples for the survey

    Parameters
    ----------
    result_path: path to store the survey data into
    sample_len: Length of the eeg sample to be displayed
    bs, fade, seperate_gans: used to identify the GAN to use
    num_imgs: the number of images for the survey
    channels: channels of the eeg data
    fs: sampling rate of the data
    """

    print("Create Survey")

    gan_folder_name = "all_subjects_" + str(sample_len) + "s_bs" + str(bs)
    gan_folder_name = gan_folder_name + "_fade" if fade else gan_folder_name
    gan_folder_name = gan_folder_name + "_separate" if seperate_gans else gan_folder_name + "_combined"
    figure_path = os.path.join(result_path, "Survey", gan_folder_name)

    gan_path = g_util.create_gan_path_from_params(result_path, sample_len, bs, fade)
    gan_path_c0 = os.path.join(gan_path, "gan_c0")
    gan_path_c1 = os.path.join(gan_path, "gan_c1")
    gan_path_comb = os.path.join(gan_path, "gan_combined")

    if seperate_gans:
        generator_c0_s5, _ = persistence.load_gan(gan_path_c0 + "/modules_stage_5.pt")
        generator_c1_s5, _ = persistence.load_gan(gan_path_c1 + "/modules_stage_5.pt")
    else:
        generator_c0_s5, _ = persistence.load_gan(gan_path_comb + "/modules_stage_5.pt")
        generator_c1_s5 = generator_c0_s5

    ds_name = "all_subjects_" + str(sample_len) + "s.dataset"
    dataset_path = os.path.join(result_path, "datasets")
    real_data_all = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)

    num_samples = 100 # len(real_data_all.train_data.X)
    classifier = g_util.get_best_base_classifier(sample_len, result_path)

    print("> Generate Samples")
    fake_pp = generate.generate_data_separate_gans_pp(generator_c0_s5, generator_c1_s5, 0, 1, num_samples,
                                                      classifier, metrics.jsd,  real_data_all, channels, fs,
                                                      clfr_th, (0, 30), jsd_th, 50)
    scale = 4
    taken_samples_real = []
    taken_samples_fake = []
    real = real_data_all.train_data.X[:num_samples]

    print("> Create Images")
    create_survey_images_raw_freq(real, fake_pp.X, num_imgs, figure_path, fs, channels, scale,
                                         str(sample_len), 2,
                                         taken_samples_real, taken_samples_fake)

# --------------------------------------------------------------------------

def create_survey_images_raw_freq(real, fake, num, figure_path, fs, channels, scale, duration, streak_limit=3,
                                  taken_samples_real = [], taken_samples_fake = []):
    """
    Create a series of images, where on the left side a time series signal is shown and on the right side
    the respective power spectral density.

    Parameters
    ----------
    real: the samples of the real data
    fake: the samples of the fake data
    num: the number of images (time + freq) to generate)
    figure_path: the path to store the images to
    fs: sampling rate of the data
    channels: the channels of the eeg data
    scale: the scale the eeg signals should be displayed
    duration: the length of the displayed eeg signal
    streak_limit: limits how long a streak of either real or fake samples can be
    taken_samples_real: list of samples from the real data that is to be excluded
    taken_samples_fake: list of samples from the fake data that is to be excluded
    """
    record = "raw freq - questions\n\n"
    if not os.path.exists(figure_path): os.makedirs(figure_path)
    figure_path = figure_path + "/"
    record_csv = ""

    num_real = 0
    num_fake = 0
    i = 0
    prev_use_real = False
    streak = 0

    while(num_real + num_fake < num):
        half = int(num/2)
        use_real = random.randint(0, 1) == 1

        if streak >= streak_limit and (prev_use_real == use_real):
            use_real = not use_real
        if num_real == half:
            use_real = False
        if num_fake == half:
            use_real = True
        if num_fake + num_real == num:
            break

        i += 1

        if use_real:
            if prev_use_real: streak += 1
            else: streak = 1

            prev_use_real = True
            num_real += 1
            sample, n = util.get_random_sample(real, taken_samples_real)

            title = str(i) + "_c" + str(random.choice(REAL_INDICES))
            record += "real (" + title + ")\n"
            record_csv += "real,"
        else:
            if not prev_use_real: streak += 1
            else: streak = 1

            prev_use_real = False
            num_fake += 1
            sample, n = util.get_random_sample(fake, taken_samples_fake)

            title = str(i) + "_c" + str(random.choice(FAKE_INDICES))
            record += "fake (" + title + ")\n"
            record_csv += "fake,"

        g_plot.plot_raw(title, sample, figure_path, True, False, fs, channels, scale, "", duration)
        img1 = Image.open(figure_path + "raw/" + title + ".png")

        left = 10
        top = 0
        right = 637
        bottom = 596
        img_res1 = img1.crop((left, top, right, bottom))

        g_plot.plot_freq(title, sample, fs, channels, figure_path, show=False, save=True, frange=(0, 40))

        img2 = Image.open(figure_path + "psd/" + title + ".png")
        left = 0
        top = 20
        right = 430
        bottom = 490
        img_res2 = img2.crop((left, top, right, bottom))

        img_res3 = util.concat_images_horizontally(img_res1, img_res2)
        # img_res3.show()

        combined_path = figure_path + "/combined/"
        if not os.path.exists(combined_path):
            os.makedirs(combined_path)
        img_res3.save(combined_path + title + ".png")

        img_res1.close()
        img_res2.close()
        img_res3.close()
        img1.close()
        img2.close()

    print("................")
    print(record)
    f = open(figure_path + "record.txt", "w+")
    f.write(record)
    f.close()

    f = open(figure_path + "record.csv", "w+")

    header = ""
    for i in range(0, num):
        header += str(i) + ","
    header = header[0: len(header) - 1] + "\n"
    record_csv = record_csv[0: len(record_csv) - 1]

    print(header + record_csv)
    f.write(header + record_csv)
    f.close()

# --------------------------------------------------------------------------
