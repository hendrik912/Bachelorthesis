import os
import torch
import joblib
import numpy as np
import eeggan.Bachelorarbeit.util as util
import eeggan.Bachelorarbeit.persistence as persistence
import eeggan.Bachelorarbeit.config as config
from eeggan.Bachelorarbeit.Baseline import BaselineV4
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from typing import Tuple
from ignite.engine import Events
from ignite.metrics import MetricUsage
from matplotlib import pyplot
from torch import Tensor, optim
from torch.utils.data import DataLoader
from eeggan.cuda import to_cuda, init_cuda
from eeggan.data.dataset import Data
from eeggan.data.preprocess.resample import downsample
from eeggan.training.handlers.plots import SpectralPlot
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.trainer import Trainer
import eeggan.Bachelorarbeit.GAN.util as g_util
from eeggan.training.handlers.metrics import WassersteinMetric, LossMetric

# --------------------------------------------------------------------------

def train_gan(parameters, result_path, gan_learning_rate, gan_epochs, fs):
    """
    Train the GANs given by the list of parameters (list of tuples)

    Parameters
    ----------
    parameters: list of tuples with param for GAN
    result_path: Path for the results
    gan_learning_rate: learning rate of gan
    gan_epochs: Number of epochs to train each stage of the gan
    fs: sampling rate of the data
    """

    # gan_path = os.path.join(result_path, "gans")

    gan_stage = 6

    for i in range(len(parameters)):
        sl = parameters[i][0]
        bs = parameters[i][1]
        fade = parameters[i][2]
        c0 = parameters[i][3]
        c1 = parameters[i][4]
        comb = parameters[i][5]
        path = g_util.create_gan_path_from_params(result_path, sl, bs, fade)

        dataset_path = os.path.join(result_path, "datasets")
        gan_path_c0 = os.path.join(path, "gan_c0")
        gan_path_c1 = os.path.join(path, "gan_c1")
        gan_path_comb = os.path.join(path, "gan_combined")

        ds_name = "all_subjects_" + str(sl) + "s.dataset"
        data_f = persistence.load_prepared_data_from_dataset(dataset_path, ds_name)


        if config.test_mode:
            print("THE SIZE OF THE DATASET IS REDUCED BECAUSE THE TESTMODE IS ENABLED (config.ini)")
            data_f.train_data.X = data_f.train_data.X[:20]
            data_f.train_data.y = data_f.train_data.y[:20]
            data_f.train_data.y_onehot = data_f.train_data.y_onehot[:20]

        data_u_0 = util.get_single_class(data_f, 0)
        data_u_1 = util.get_single_class(data_f, 1)

        lr_g = gan_learning_rate

        if comb:
            print("\n> train GAN both classes")
            train_attention_model(data_f, gan_path_comb, fs=fs, epochs=gan_epochs, batch_size=bs,
                                        n_progressive_stages=gan_stage, use_fading=fade, learning_rate=lr_g)
        else:
            if c0:
                print("\n> train GAN class 0")
                train_attention_model(data_u_0, gan_path_c0, fs=fs, epochs=gan_epochs, batch_size=bs,
                                        n_progressive_stages=gan_stage, use_fading=fade, learning_rate=lr_g)
            if c1:
                print("\n> train GAN class 1")
                train_attention_model(data_u_1, gan_path_c1, fs=fs, epochs=gan_epochs, batch_size=bs,
                                        n_progressive_stages=gan_stage, use_fading=fade, learning_rate=lr_g)

# --------------------------------------------------------------------------

def train_attention_model(dataset, model_path, fs, epochs = 2000, batch_size=128, n_progressive_stages = 5, use_fading = False, learning_rate=0.001):
    """
    Creates the GAN config and the model builder and prepares the models for the training
    """

    n_epochs_per_stage = epochs

    input_length = len(dataset.train_data.X[0][0])
    config = create_config(fs, input_length, n_progressive_stages, n_epochs_per_stage, n_chans=11, batch_size=batch_size, use_fading=use_fading, lr_g=learning_rate)
    model_builder = create_model_builder(config)

    os.makedirs(model_path, exist_ok=True)

    joblib.dump(config, os.path.join(model_path, 'config.dict'), compress=False)
    joblib.dump(model_builder, os.path.join(model_path, 'model_builder.jblb'), compress=True)

    # create discriminator and generator modules
    discriminator = model_builder.build_discriminator()
    generator = model_builder.build_generator()

    # initiate weights
    generator.apply(weight_filler)
    discriminator.apply(weight_filler)

    # trainer engine
    trainer = GanSoftplusTrainer(10, discriminator, generator, config['r1_gamma'], config['r2_gamma'])

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, config['n_stages'], config['use_fade'],
                                             config['n_epochs_fade'], freeze_stages=config['freeze_stages'])
    progression_handler.set_progression(0, 1.)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

    # Sets the module in training mode.
    generator.train()
    discriminator.train()

    train(dataset.train_data, model_path, progression_handler, trainer, config['n_batch'],
          config['lr_d'], config['lr_g'], config['betas'], config['n_epochs_per_stage'], config['n_epochs_metrics'],
          config['plot_every_epoch'], config['orig_fs'])

# --------------------------------------------------------------------------

def train(train_data, result_path: str, progression_handler: ProgressionHandler, trainer: Trainer, n_batch: int,
          lr_d: float, lr_g: float, betas: Tuple[float, float], n_epochs_per_stage: int, n_epochs_metrics: int,
          plot_every_epoch: int, orig_fs: float):
    """
    The actual training happens in this function
    """

    plot_path = os.path.join(result_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    init_cuda()

    discriminator = progression_handler.discriminator
    generator = progression_handler.generator
    discriminator, generator = to_cuda(discriminator, generator)

    # usage to update every epoch and compute once at end of stage
    usage_metrics = MetricUsage(Events.STARTED, Events.EPOCH_COMPLETED(every=n_epochs_per_stage),
                                Events.EPOCH_COMPLETED(every=n_epochs_metrics))

    for stage in range(progression_handler.n_stages):

        # optimizer
        optim_discriminator = optim.Adam(progression_handler.get_trainable_discriminator_parameters(), lr=lr_d,
                                         betas=betas)
        optim_generator = optim.Adam(progression_handler.get_trainable_generator_parameters(), lr=lr_g, betas=betas)
        trainer.set_optimizers(optim_discriminator, optim_generator)

        # scale data for current stage
        sample_factor = 2 ** (progression_handler.n_stages - stage - 1)
        X_block = downsample(train_data.X, factor=sample_factor, axis=2)

        # initiate spectral plotter
        spectral_plot = SpectralPlot(pyplot.figure(), plot_path, "spectral_stage_%d_" % stage, X_block.shape[2],
                                     orig_fs / sample_factor)

        event_name = Events.EPOCH_COMPLETED(every=plot_every_epoch)
        spectral_handler = trainer.add_event_handler(event_name, spectral_plot)

        # initiate metrics
        metric_wasserstein = WassersteinMetric(100, np.prod(X_block.shape[1:]).item())
        metric_loss = LossMetric()

        metrics = [metric_wasserstein, metric_loss]
        metric_names = ["wasserstein", "loss"]

        trainer.attach_metrics(metrics, metric_names, usage_metrics)

        # wrap into cuda loader
        train_data_tensor: Data[Tensor] = Data(
            *to_cuda(Tensor(X_block), Tensor(train_data.y), Tensor(train_data.y_onehot)))
        train_loader = DataLoader(train_data_tensor, batch_size=n_batch, shuffle=True)

        # train stage
        state = trainer.run(train_loader, (stage + 1) * n_epochs_per_stage)
        trainer.remove_event_handler(spectral_plot, event_name)

        # modules to save
        to_save = {'discriminator': discriminator, 'generator': generator,
                   'optim_discriminator': optim_discriminator, 'optim_generator': optim_generator}

        # save stuff
        torch.save(to_save, os.path.join(result_path, 'modules_stage_%d.pt' % stage))
        torch.save(dict([(name, to_save[name].state_dict()) for name in to_save.keys()]),
                   os.path.join(result_path, 'states_stage_%d.pt' % stage))
        torch.save(trainer.state.metrics, os.path.join(result_path, 'metrics_stage_%d.pt' % stage))

        # advance stage if not last
        trainer.detach_metrics(metrics, usage_metrics)
        if stage != progression_handler.n_stages - 1:
            progression_handler.advance_stage()

# --------------------------------------------------------------------------

def create_config(fs, input_length, n_progressive_stages, n_epoches_per_stage, n_chans, batch_size, use_fading, lr_g):
    """
    Create the GAN-config

    Parameters
    ----------
    fs: Sampling rate of the data
    input_length: Sample length in seconds
    n_progressive_stages: Number of stages
    n_epoches_per_stage: Number of epochs per stage
    n_chans: Number of EEG channels used in the data
    batch_size: the batch size
    use_fading: Whether or not to use fading
    lr_g: the learning rate

    Returns
    -------
    Dictionary with param for GAN
    """

    return dict(
        n_chans=n_chans,  # number of channels in data
        n_classes=2,  # number of classes in data
        orig_fs=fs,  # sampling rate of data

        n_batch=batch_size, # 32, # 64, # 128,  # batch size
        n_stages=n_progressive_stages,  # number of progressive stages
        n_epochs_per_stage=n_epoches_per_stage,  # epochs in each progressive stage
        n_epochs_metrics=100,
        plot_every_epoch=100,
        n_epochs_fade=int(0.1 * n_epoches_per_stage),
        use_fade=use_fading,
        freeze_stages=True,

        n_latent=200,  # latent vector size
        r1_gamma=10.,
        r2_gamma=0.,
        lr_d=0.005,  # discriminator learning rate
        lr_g=lr_g, # 0.001,  # generator learning rate
        betas=(0., 0.99),  # optimizer betas

        n_filters=120,
        n_time=input_length,

        upsampling='conv',
        downsampling='conv',
        discfading='cubic',
        genfading='cubic',
    )

# --------------------------------------------------------------------------

def create_model_builder(config):
    """
    Create the baseline model builder
    """

    return BaselineV4(config['n_stages'], config['n_latent'], config['n_time'],
                      config['n_chans'], config['n_classes'], config['n_filters'],
                      upsampling=config['upsampling'], downsampling=config['downsampling'],
                      discfading=config['discfading'], genfading=config['genfading'])

# --------------------------------------------------------------------------


