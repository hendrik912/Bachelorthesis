# This file loads all the parameters from the config file
from distutils.util import strtobool
import eeggan.Bachelorarbeit.persistence as persistence

# -----------------------------------------------------------------------

config =         persistence.load_config()
classifierInfo = config["CLASSIFIER"]
ganInfo =        config["GAN"]
dataInfo =       config["DATA"]
surveyInfo =     config["SURVEY"]
evalInfo =       config["EVALUATION"]
generalInfo =    config["GENERAL"]
pl_stepsInfo =   config["PIPELINE_STEPS"]

test_mode = bool(strtobool(generalInfo["test_mode"]))

# -----------------------------------------------------------------------
# Pipeline Steps

create_datasets_b =        bool(strtobool(pl_stepsInfo["create_datasets"]))
train_gan_b =              bool(strtobool(pl_stepsInfo["train_gans"]))
train_base_classifiers_b = bool(strtobool(pl_stepsInfo["train_base_clfr"]))
train_augm_classifiers_b = bool(strtobool(pl_stepsInfo["train_augm_clfr"]))
create_survey_b =          bool(strtobool(pl_stepsInfo["create_survey"]))
eval_gans_b =              bool(strtobool(pl_stepsInfo["eval_gans"]))
eval_survey_b =            bool(strtobool(pl_stepsInfo["eval_survey"]))
eval_classifiers_b =       bool(strtobool(pl_stepsInfo["eval_classifiers"]))

# -----------------------------------------------------------------------
# Datasets

data_filter = bool(strtobool(dataInfo["filter"]))
data_lpass =  int(dataInfo["lpass"])
data_hpass =  int(dataInfo["hpass"])

# -----------------------------------------------------------------------
# Gan Parameters

gan_bs =            int(ganInfo["batch_size"])
gan_epochs =        int(ganInfo["epochs"])
gan_stage =         int(ganInfo["stage"])
gan_fading =        bool(strtobool(ganInfo["use_fading"]))
gan_learning_rate = float(ganInfo["learning_rate"])

# -----------------------------------------------------------------------
# Classifier Parameters

c_pred_threshold =    float(classifierInfo["pred_th"])
c_jsd_threshold =     float(classifierInfo["jsd_th"])
c_use_jsd_threshold = bool(strtobool(classifierInfo["use_jsd_th"]))
c_lr =                float(classifierInfo["learning_rate"])
c_filter_range =      float(classifierInfo["hpass"]), float(classifierInfo["lpass"])
c_batch_size =        int(classifierInfo["batch_size"])
c_epochs =            int(classifierInfo["epochs"])
c_iterations =        int(classifierInfo["iterations"])
c_ratio =             int(classifierInfo["ratio_end"])
c_use_clfr_th =       bool(strtobool(classifierInfo["use_clfr_th"]))
c_ratio_start =       int(classifierInfo["ratio_start"])

# -----------------------------------------------------------------------
# Survey Parameters

s_sample_len =    int(surveyInfo["sample_len"])
s_bs =            int(surveyInfo["batch_size"])
s_num_imgs =      int(surveyInfo["num_imgs"])
s_fading =        bool(strtobool(surveyInfo["use_fading"]))
s_separate_gans = bool(strtobool(surveyInfo["separate_gans"]))
s_pred_threshold =    float(classifierInfo["pred_th"])
s_jsd_threshold =     float(classifierInfo["jsd_th"])

# -----------------------------------------------------------------------
# Evaluation Parameters

eval_g_calc_metrics_b = bool(strtobool(evalInfo["gan_calc_metrics"]))
eval_c_calc_metrics_b = bool(strtobool(evalInfo["c_calc_metrics"]))
eval_num_samples =      int(evalInfo["num_samples"])
eval_times =            int(evalInfo["times"])

# -----------------------------------------------------------------------