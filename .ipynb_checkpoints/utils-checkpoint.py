from functools import partial
import tabpfn.encoders as encoders

from tabpfn.transformer import TransformerModel
from tabpfn.utils import get_uniform_single_eval_pos_sampler
import torch
import math
from tabpfn_new.scripts.model_builder import *


# spoof of the tabpfn get_model method to get dataloader, much easier than to reconstruct that process...
def get_dataloader(config, device, should_train=True, verbose=False, state_dict=None, epoch_callback=None):
    import tabpfn_new.priors as priors
    from tabpfn.train import train, Losses
    extra_kwargs = {}
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    if 'aggregate_k_gradients' not in config or config['aggregate_k_gradients'] is None:
        config['aggregate_k_gradients'] = math.ceil(config['batch_size'] * ((config['nlayers'] * config['emsize'] * config['bptt'] * config['bptt']) / 10824640000))

    config['num_steps'] = math.ceil(config['num_steps'] * config['aggregate_k_gradients'])
    config['batch_size'] = math.ceil(config['batch_size'] / config['aggregate_k_gradients'])
    config['recompute_attn'] = config['recompute_attn'] if 'recompute_attn' in config else False

    def make_get_batch(model_proto, **extra_kwargs):
        def new_get_batch(batch_size, seq_len, num_features, hyperparameters
                , device, model_proto=model_proto
                , **kwargs):
            kwargs = {**extra_kwargs, **kwargs} # new args overwrite pre-specified args
            return model_proto.get_batch(
                batch_size=batch_size
                , seq_len=seq_len
                , device=device
                , hyperparameters=hyperparameters
                , num_features=num_features, **kwargs)
        return new_get_batch

    if config['prior_type'] == 'prior_bag':
        # Prior bag combines priors
        get_batch_gp = make_get_batch(priors.fast_gp)
        get_batch_mlp = make_get_batch(priors.mlp)
        if 'flexible' in config and config['flexible']:
            get_batch_gp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_gp})
            get_batch_mlp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_mlp})
        prior_bag_hyperparameters = {'prior_bag_get_batch': (get_batch_gp, get_batch_mlp)
            , 'prior_bag_exp_weights_1': 2.0}
        prior_hyperparameters = {**get_mlp_prior_hyperparameters(config), **get_gp_prior_hyperparameters(config)
            , **prior_bag_hyperparameters}
        model_proto = priors.prior_bag
    else:
        if config['prior_type'] == 'mlp':
            prior_hyperparameters = get_mlp_prior_hyperparameters(config)
            model_proto = priors.mlp
        elif config['prior_type'] == 'gp':
            prior_hyperparameters = get_gp_prior_hyperparameters(config)
            model_proto = priors.fast_gp
        elif config['prior_type'] == 'gp_mix':
            prior_hyperparameters = get_gp_mix_prior_hyperparameters(config)
            model_proto = priors.fast_gp_mix
        elif config['prior_type'] == 'forest':
            prior_hyperparameters = get_forest_prior_hyperparameters(config)
            model_proto = priors.forest
        else:
            raise Exception()

        if 'flexible' in config and config['flexible']:
            get_batch_base = make_get_batch(model_proto)
            extra_kwargs['get_batch'] = get_batch_base
            model_proto = priors.flexible_categorical

    if config.get('flexible'):
        prior_hyperparameters['normalize_labels'] = True
        prior_hyperparameters['check_is_compatible'] = True
    prior_hyperparameters['prior_mlp_scale_weights_sqrt'] = config[
        'prior_mlp_scale_weights_sqrt'] if 'prior_mlp_scale_weights_sqrt' in prior_hyperparameters else None
    prior_hyperparameters['rotate_normalized_labels'] = config[
        'rotate_normalized_labels'] if 'rotate_normalized_labels' in prior_hyperparameters else True

    use_style = False

    if 'differentiable' in config and config['differentiable']:
        get_batch_base = make_get_batch(model_proto, **extra_kwargs)
        extra_kwargs = {'get_batch': get_batch_base, 'differentiable_hyperparameters': config['differentiable_hyperparameters']}
        model_proto = priors.differentiable_prior
        use_style = True
    print(f"Using style prior: {use_style}")

    if (('nan_prob_no_reason' in config and config['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config and config['nan_prob_a_reason'] > 0.0) or
        ('nan_prob_unknown_reason' in config and config['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    if config['max_num_classes'] == 2:
        loss = Losses.bce
    elif config['max_num_classes'] > 2:
        loss = Losses.ce(config['max_num_classes'])


    check_is_compatible = False if 'multiclass_loss_type' not in config else (config['multiclass_loss_type'] == 'compatible')
    config['multiclass_type'] = config['multiclass_type'] if 'multiclass_type' in config else 'rank'
    config['mix_activations'] = config['mix_activations'] if 'mix_activations' in config else False

    config['bptt_extra_samples'] = config['bptt_extra_samples'] if 'bptt_extra_samples' in config else None
    config['eval_positions'] = [int(config['bptt'] * 0.95)] if config['bptt_extra_samples'] is None else [int(config['bptt'])]

    epochs = 0 if not should_train else config['epochs']
    #print('MODEL BUILDER', model_proto, extra_kwargs['get_batch'])
    extra_prior_kwargs_dict={
            'num_features': config['num_features']
            , 'hyperparameters': prior_hyperparameters
            #, 'dynamic_batch_size': 1 if ('num_global_att_tokens' in config and config['num_global_att_tokens']) else 2
            , 'batch_size_per_gp_sample': config.get('batch_size_per_gp_sample', None)
    }
    return model_proto.DataLoader, extra_prior_kwargs_dict