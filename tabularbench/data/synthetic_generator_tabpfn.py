
import sys

import torch
from tabpfn.notebook_utils import *
from tabpfn.priors.utils import uniform_int_sampler_f
from tabpfn.scripts.model_builder import get_model
from tabpfn.scripts.model_configs import *


def synthetic_dataset_generator_tabpfn(
        n_samples: int,
        min_features: int,
        max_features: int,
        max_classes: int
    ):

    config = get_prior_config_causal(max_features=max_features)

    config['prior_type'] = 'mlp'
    config['differentiable'] = True
    config['flexible'] = True

    config['num_classes'] = uniform_int_sampler_f(2, max_classes)
    config['balanced'] = False

    config['bptt_extra_samples'] = None
    config['num_features_used'] = {
        'uniform_int_sampler_f(3, max_features)': uniform_int_sampler_f(min_features, max_features)
    }
    # diff
    config['output_multiclass_ordered_p'] = 0.
    del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

    config['multiclass_type'] = 'rank'
    del config['differentiable_hyperparameters']['multiclass_type']

    config['sampling'] = 'normal' # vielleicht schlecht?
    del config['differentiable_hyperparameters']['sampling']

    config['pre_sample_causes'] = True
    # end diff

    config['multiclass_loss_type'] = 'nono' # 'compatible'
    config['normalize_to_ranking'] = False # False

    config['categorical_feature_p'] = .2 # diff: .0

    # turn this back on in a random search!?
    config['nan_prob_no_reason'] = .0
    config['nan_prob_unknown_reason'] = .0 # diff: .0
    config['set_value_to_nan'] = .1 # diff: 1.

    config['normalize_with_sqrt'] = False

    config['new_mlp_per_example'] = True
    config['prior_mlp_scale_weights_sqrt'] = True
    config['batch_size_per_gp_sample'] = None

    config['normalize_ignore_label_too'] = False

    config['differentiable_hps_as_style'] = False

    config['random_feature_rotation'] = True
    config['rotate_normalized_labels'] = True

    config["mix_activations"] = False # False heisst eig True

    config['emsize'] = 512
    config['nhead'] = config['emsize'] // 128
    config['bptt'] = n_samples
    config['canonical_y_encoder'] = False

    config['aggregate_k_gradients'] = 1
    config['batch_size'] = 1
    config['num_steps'] = 2**63
    config['epochs'] = 400
    config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

    config['train_mixed_precision'] = True
    config['efficient_eval_masking'] = True

    config['normalize_by_used_features'] = False

    config['min_eval_pos'] = n_samples // 2
    config['max_eval_pos'] = n_samples // 2 + 1


    config_sample = evaluate_hypers(config)

    config_sample['batch_size'] = 1

    with DisablePrinting():
        model = get_model(config_sample, 'cpu', should_train=False, verbose=0)

    data_iter = iter(model[3])

    for (_, data, _), targets, _ in data_iter:
     
        x = data[:, 0, :]
        y = targets

        if torch.all(y == -100):
            # in case of too many classes, the synthetic generator is not able to split the dataset
            # in a way that the training and validation set have the same number of classes
            # the generator returns -100 as a label for all observations in this case
            continue
        
        # remove all zero columns
        x = x[:, x.sum(dim=0) != 0]

        if x.shape[1] < min_features:
            continue

        if torch.isnan(x).sum() or torch.isnan(y).sum():
            continue

        x = x.numpy()
        y = y.numpy()

        yield x, y


class DisablePrinting:

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout


if __name__  == '__main__':

    gen = synthetic_dataset_generator_tabpfn(
        min_samples = 100,
        max_samples = 10000,
        min_features = 3,
        max_features = 133,
        max_classes = 10
    )
    x, y = next(gen)
    pass