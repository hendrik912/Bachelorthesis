from eeggan.Bachelorarbeit.Classifier.ModelType import ModelType
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F

# --------------------------------------------------------------------------

def create_classifier(in_chans, n_classes, model_type=ModelType.SHALLOW, final_conv_length = 'auto',
                      cuda = True, lr = 0.01 * 0.01, weight_decay=0.0005, input_time_length = None):
    """
    Create a classifier of the given model type and parameters

    Parameters
    ----------
    in_chans: number of channels of the eeg data
    n_classes: number of classes of the data
    model_type: either Shallow FBSCPNet or Deep4Net
    cuda: Whether to use Cuda or not
    lr: learning rate

    Returns
    -------
    the model

    """
    if model_type == ModelType.SHALLOW:
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length)
    elif model_type == ModelType.DEEP:
        model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                         input_time_length=input_time_length,
                         final_conv_length=final_conv_length)
    if cuda:
        model.cuda()

    optimizer = AdamW(model.parameters(), lr= lr, weight_decay=weight_decay)
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True)
    return model

# --------------------------------------------------------------------------