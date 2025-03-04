import os
import itertools
import argparse
import time
import datetime
import yaml
import pickle
import pathlib
from contextlib import nullcontext


import torch
from torch import nn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import tabpfn.utils as utils
from tabpfn_new.transformer import TransformerModel
from tabpfn.utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler
import tabpfn.priors as priors
import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings
from tabpfn.utils import init_dist
from torch.cuda.amp import autocast, GradScaler
from torch import nn


class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    def ce_weighted(num_classes, weight):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=weight.float())
    bce = nn.BCEWithLogitsLoss(reduction='none')

# stratified split, such that the class distribution is roughly equal in train and test (before/after split point)
def balance_split(data, targets, pos):
    #print(data.shape, targets.shape)
    #pos = max(min(data.shape[0]-2**targets.shape[-1]-1, pos), 2**targets.shape[-1])
    new_X, new_y = [], []
    for i in range(targets.shape[1]):
        #print(torch.unique(targets[:,1], return_counts=True))
        data_b, targets_b = data[:,i,:].cpu(), targets[:,i].cpu()
        if len(torch.unique(targets_b, return_counts=True)[1])>1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size = data.shape[0]-pos)
            sss.get_n_splits(data_b,targets_b)
            
            train_index, test_index= next(sss.split(data_b,targets_b))
            X_train, y_train, X_test, y_test = data_b[train_index], targets_b[train_index], data_b[test_index], targets_b[test_index]
        
            X = torch.unsqueeze(torch.cat((X_train,X_test), dim=0), dim=1)
            y = torch.unsqueeze(torch.cat((y_train,y_test), dim=0), dim=1)
            new_X.append(X)
            new_y.append(y)
        else:
            new_X.append(torch.unsqueeze(data_b, dim=1))
            new_y.append(torch.unsqueeze(targets_b, dim=1))
    X = torch.cat((*new_X,), dim=1).to(data.device)
    y = torch.cat((*new_y,), dim=1).to(data.device)
    return X, y


def train(priordataloader_class, criterion, encoder_generator, emsize=200, nhid=200, nlayers=6, nhead=2, dropout=0.0,
          epochs=10, steps_per_epoch=100, batch_size=200, bptt=10, lr=None, weight_decay=0.0, warmup_epochs=10, input_normalization=False,
          y_encoder_generator=None, pos_encoder_generator=None, decoder=None, extra_prior_kwargs_dict={}, scheduler=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, validation_period=10, single_eval_pos_gen=None, bptt_extra_samples=None, gpu_device='cuda:0',
          aggregate_k_gradients=1, verbose=True, style_encoder_generator=None, epoch_callback=None,
          initializer=None, initialize_with_model=None, train_mixed_precision=False, efficient_eval_masking=True, 
          microbiome_test=False, config=None, **model_extra_args
          ):
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    #print(f'Using {device} device')
    using_dist, rank, device = init_dist(device)
    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples:
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt
    dl = priordataloader_class(num_steps=steps_per_epoch, batch_size=batch_size, eval_pos_seq_len_sampler=eval_pos_seq_len_sampler, seq_len_maximum=bptt+(bptt_extra_samples if bptt_extra_samples else 0), device=device, **extra_prior_kwargs_dict)

    encoder = encoder_generator(dl.num_features, emsize) if not config.get("no_encoder", False) else None
    #style_def = dl.get_test_batch()[0][0] # the style in batch of the form ((style, x, y), target, single_eval_pos)
    style_def = None
    #print(f'Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}')
    style_encoder = style_encoder_generator(style_def.shape[1], emsize) if (style_def is not None) else None
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1
    model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout, style_encoder=style_encoder,
                             y_encoder=y_encoder_generator(1, emsize), input_normalization=input_normalization,
                             pos_encoder=(pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, bptt*2),
                             decoder=decoder, init_method=initializer, efficient_eval_masking=efficient_eval_masking, **model_extra_args
                             )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    #print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    try:
        for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    dl.model = model


    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        accs = []
        class_pred_measure = []
        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                
                if bptt_extra_samples is None:
                    single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                else:
                    single_eval_pos = targets.shape[0] - bptt_extra_samples
                #print(targets.shape, data[1].shape)
                # If style is set to None, it should not be transferred to device
                new_data, targets = balance_split(data[1], targets, single_eval_pos)
                data = (data[0], new_data, targets)
                output = model(tuple(e.to(device) if torch.is_tensor(e) else e for e in data) if isinstance(data, tuple) else data.to(device)
                               , single_eval_pos=single_eval_pos)

                forward_time = time.time() - before_forward
                all_targets = targets
                if single_eval_pos is not None:
                    targets = targets[single_eval_pos:]
                if isinstance(criterion, nn.GaussianNLLLoss):
                    assert output.shape[-1] == 2, \
                        'need to write a little bit of code to handle multiple regression targets at once'
                    mean_pred = output[..., 0]
                    var_pred = output[..., 1].abs()
                    losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
                elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                    losses = criterion(output.flatten(), targets.to(device).flatten())
                elif isinstance(criterion, nn.CrossEntropyLoss):
                    if config["weight_classes"]:
                        #weights = (torch.unique(all_targets, return_counts=True)[1]*2/torch.unique(all_targets, return_counts=True)[1]).flip(dims=(0,)).to(device)
                        weights = (torch.unique(all_targets, return_counts=True)[1]/torch.sum(torch.unique(all_targets, return_counts=True)[1])).flip(dims=(0,)).to(device)
                        #print(weights)
                        #print(torch.unique(targets, return_counts=True)[1][-1]/targets.shape[0])
                        if len(weights)!=2:
                            weights = torch.tensor([1,1]).to(device)
                        #weights = torch.nn.functional.softmax(weights.float(), dim=-1)*2
                        weights = weights+config["weight_increase"]
                        #weights = weights*2
                        criterion_new = Losses.ce_weighted(n_out,weights)
                        losses = criterion_new(output.reshape(-1, n_out), targets.to(device).long().flatten())
                    else:
                        losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                else:
                    losses = criterion(output, targets)
                #print(output[:3], targets[:3])
                losses = losses.view(*output.shape[0:2])
                loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                loss = loss / aggregate_k_gradients

                if scaler: loss = scaler.scale(loss)
                loss.backward()
                
                preds = torch.argmax(output, dim=-1).float() if isinstance(criterion, nn.CrossEntropyLoss) else (torch.nn.functional.sigmoid(output)>0.5).float()[:,:,0]
                #print(torch.sum(preds).dtype, torch.sum(targets).dtype)
                accuracy = torch.mean((preds==targets)[targets!=-100].float())
                accs.append(accuracy)
                class_pred = torch.sum(preds, dim=0)/output.shape[0]
                class_tgt = torch.sum(targets, dim=0)/output.shape[0]
                class_pred_measure.append(torch.mean((class_pred-class_tgt)*torch.sign(class_tgt-0.5)))
                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:                            
                    #print(torch.unique(targets))
                    #print("\nLast output: ", output[:10])
                    print("\n\n% Positive predictions:")
                    for elem in (class_pred): print(f"{elem:2.3f}  ", end='') 
                    print("\n% Positive targets:")
                    for elem in (torch.sum(targets, dim=0)/targets.shape[0]): print(f"{elem:2.3f}  ", end='') 
                    print(f"\nTrain sample accuracy: {accuracy.item():2.3f}")
                    if scaler: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    try:
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    optimizer.zero_grad()

                step_time = time.time() - before_forward

                if not torch.isnan(loss):
                    total_loss += losses.mean().cpu().detach().item()
                    total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*\
                        losses[:bptt-single_eval_pos].mean().cpu().detach()

                    total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                nan_steps += nan_share
                ignore_steps += (targets == -100).float().mean()

            before_get_batch = time.time()
        #return total_loss / steps_per_epoch, (total_positional_losses / total_positional_losses_recorded).tolist(),\
        return total_loss / steps_per_epoch, torch.nanmean(torch.tensor(accs)), torch.mean(torch.tensor(class_pred_measure)), 0, \
            time_to_get_batch, forward_time, step_time, nan_steps.cpu().item()/(batch+1),\
               ignore_steps.cpu().item()/(batch+1)

    
    total_loss = float('inf')
    total_positional_losses = float('inf')
    if epochs>0:
        ### logging
        if "run_name" not in config or config["run_name"]=="time":
            run_name = time.strftime("%Y%m%d-%H%M%S")
        else:
            run_name = config["run_name"]
        dir_path = os.path.abspath(os.getcwd())
        path = dir_path + f"/logs/trainrun_{run_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/config.pkl', 'wb') as f:
            pickle.dump(config, f)    
        with open(path + '/config.txt', 'w') as f:
            for k, v in config.items():
                f.write(str(k) + ' >>> '+ str(v) + '\n\n')
        epoch_losses = []
        accs = []
        class_preds = []
        mb_results = None
        try:
            for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
                epoch_start_time = time.time()
                if extra_prior_kwargs_dict["hyperparameters"]["multiclass_type"] == "variable_balance":
                    extra_prior_kwargs_dict["hyperparameters"]["epoch_frac"] = (epoch/epochs)**2
                    dl = priordataloader_class(num_steps=steps_per_epoch, batch_size=batch_size, eval_pos_seq_len_sampler=eval_pos_seq_len_sampler, seq_len_maximum=bptt+(bptt_extra_samples if bptt_extra_samples else 0), device=device, **extra_prior_kwargs_dict)
                    dl.model = model
                total_loss, acc, class_pred, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                    train_epoch()
                epoch_losses.append(total_loss)
                accs.append(acc)
                class_preds.append(class_pred)
                if microbiome_test:
                    with torch.no_grad():
                        results = mb_test(model, config, device)
                    mb_results = torch.tensor(results.values) if mb_results is None else torch.cat((mb_results, torch.tensor(results.values)), dim=0)
                if hasattr(dl, 'validate') and epoch % validation_period == 0:
                    with torch.no_grad():
                        val_score = dl.validate(model)
                else:
                    val_score = None
    
                if verbose:
                    print('-' * 89)
                    print(
                        f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                        #f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                        f' mean accuracy {acc:5.4f} | '
                        f' preds imbalance measure {class_pred:5.2f} | '
                        f' lr {scheduler.get_last_lr()[0]} | '
                        f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                        f' forward time {forward_time:5.2f}' 
                        f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                        + (f'val score {val_score}' if val_score is not None else ''))
                    print('-' * 89)
    
                # stepping with wallclock time based scheduler
                if epoch_callback is not None and rank == 0:
                    epoch_callback(model, epoch / epochs)
                scheduler.step()
        except KeyboardInterrupt:
            pass
        torch.save(torch.tensor(epoch_losses), path + "/losses")
        torch.save(torch.tensor(accs), path + "/accuracies")
        torch.save(torch.tensor(class_preds), path + "/class_preds")
        torch.save(mb_results, path + "/mb_results")
    if rank == 0: # trivially true for non-parallel training
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        return total_loss, total_positional_losses, model.to('cpu'), dl

import os
import sys
#print(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from data_prep_utils import *
from evaluate import *
from tabpfn_new.scripts.transformer_prediction_interface import TabPFNClassifier
from tabpfn_new.scripts.model_builder import save_model

mb_data, mb_labels = get_microbiome(path="datasets/data_all.csv")
mb_data = remove_zero_features(mb_data)
#mb_data = top_anova(mb_data, mb_labels)
mb_data, mb_labels = unison_shuffled_copies(mb_data, mb_labels, seed=42)
def mb_test(model, config, device):
    seed=42
    pred_model = TabPFNClassifier(model, config, device=device, no_preprocess_mode=False, norm=config["normalize"], clr=config["clr"])
    metrics = ["accuracy", "precision", "recall", "roc_auc", "f1"]
    results = pd.DataFrame(np.zeros((1, len(metrics)+1)), index=["Medical TabPFN"], columns=metrics+["runtime"])
    results[:],_ = cross_validate_sample(pred_model, mb_data, mb_labels, metrics, strat_split=True, cv=4, sampling=None, 
                                         reducer=AnovaSelect(), max_samples=1024, seed=seed)
    data = top_anova(mb_data, mb_labels)
    X_train, X_test, y_train, y_test = train_test_split(data, mb_labels, train_size=1024, test_size=200, random_state=seed)
    X_train, y_train = reduce_n_samples(X_train, y_train, 1024)
    X_train, y_train = unison_shuffled_copies(X_train, y_train)
    with torch.no_grad():
        pred_model.fit(X_train, y_train)
        preds = pred_model.predict(X_test)
    print("\n% of positive predictions: ", np.sum(preds)/preds.shape[0])
    print(results)
    return results

def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
    config_parser.add_argument('--config')
    parser = argparse.ArgumentParser()
    parser.add_argument('prior')
    parser.add_argument('--loss_function', default='barnll')
    # Optional Arg's for `--loss_function barnll`
    parser.add_argument('--min_y', type=float, help='barnll can only model y in strict ranges, this is the minimum y can take.')
    parser.add_argument('--max_y', type=float, help='barnll can only model y in strict ranges, this is the maximum y can take.')
    parser.add_argument('--num_buckets', default=100, type=int)
    #parser.add_argument('--num_features', default=None, type=int, help='Specify depending on the prior.')
    parser.add_argument("--extra_prior_kwargs_dict", default={}, dest="extra_prior_kwargs_dict", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", help='Specify depending on the prior.')
    parser.add_argument('--encoder', default='linear', type=str, help='Specify depending on the prior.')
    parser.add_argument('--y_encoder', default='linear', type=str, help='Specify depending on the prior. You should specify this if you do not fuse x and y.')
    parser.add_argument('--pos_encoder', default='none', type=str, help='Specify depending on the prior.')
    parser.add_argument('--bptt', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--validation_period', default=10, type=int)
    parser.add_argument('--permutation_invariant_max_eval_pos', default=None, type=int, help='Set this to an int to ')
    parser.add_argument('--permutation_invariant_sampling', default='weighted', help="Only relevant if --permutation_invariant_max_eval_pos is set.")
    parser.add_argument('--train_mixed_precision', action='store_true')

    # these can likely be mostly left at defaults
    parser.add_argument('--emsize', default=512, type=int) # sometimes even larger is better e.g. 1024
    parser.add_argument('--nlayers', default=6, type=int)
    parser.add_argument('--nhid', default=None, type=int) # 2*emsize is the default
    parser.add_argument('--nhead', default=4, type=int) # nhead = emsize / 64 in the original paper
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', '--learning_rate', default=.001, type=float) # try also .0003, .0001, go lower with lower batch size

    args, _ = _parse_args(config_parser, parser)

    if args.nhid is None:
        args.nhid = 2*args.emsize

    prior = args.__dict__.pop('prior')

    if prior == 'gp':
        prior = priors.fast_gp.DataLoader
    elif prior == 'ridge':
        prior = priors.ridge.DataLoader
    elif prior == 'stroke':
        prior = priors.stroke.DataLoader
    elif prior == 'mix_gp':
        prior = priors.fast_gp_mix.DataLoader
    else:
        raise NotImplementedError(f'Prior == {prior}.')

    loss_function = args.__dict__.pop('loss_function')

    criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    classificiation_criterion = nn.CrossEntropyLoss(reduction='none')
    num_buckets = args.__dict__.pop('num_buckets')
    max_y = args.__dict__.pop('max_y')
    min_y = args.__dict__.pop('min_y')
    # criterion = nn.MSELoss(reduction='none')

    if loss_function == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='none')
    elif loss_function == 'gaussnll':
        criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    elif loss_function == 'mse':
        criterion = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f'loss_function == {loss_function}.')



    encoder = args.__dict__.pop('encoder')
    y_encoder = args.__dict__.pop('y_encoder')

    def get_encoder_generator(encoder):
        if encoder == 'linear':
            encoder_generator = encoders.Linear
        elif encoder == 'mlp':
            encoder_generator = encoders.MLP
        elif encoder == 'positional':
            encoder_generator = encoders.Positional
        else:
            raise NotImplementedError(f'A {encoder} encoder is not valid.')
        return encoder_generator

    encoder_generator = get_encoder_generator(encoder)
    y_encoder_generator = get_encoder_generator(y_encoder)

    pos_encoder = args.__dict__.pop('pos_encoder')

    if pos_encoder == 'none':
        pos_encoder_generator = None
    elif pos_encoder == 'sinus':
        pos_encoder_generator = positional_encodings.PositionalEncoding
    elif pos_encoder == 'learned':
        pos_encoder_generator = positional_encodings.LearnedPositionalEncoding
    elif pos_encoder == 'paired_scrambled_learned':
        pos_encoder_generator = positional_encodings.PairedScrambledPositionalEncodings
    else:
        raise NotImplementedError(f'pos_encoer == {pos_encoder} is not valid.')

    permutation_invariant_max_eval_pos = args.__dict__.pop('permutation_invariant_max_eval_pos')
    permutation_invariant_sampling = args.__dict__.pop('permutation_invariant_sampling')
    if permutation_invariant_max_eval_pos is not None:
        if permutation_invariant_sampling == 'weighted':
            get_sampler = get_weighted_single_eval_pos_sampler
        elif permutation_invariant_sampling == 'uniform':
            get_sampler = get_uniform_single_eval_pos_sampler
        else:
            raise ValueError()
        args.__dict__['single_eval_pos_gen'] = get_sampler(permutation_invariant_max_eval_pos)


    print("ARGS for `train`:", args.__dict__)

    train(prior, criterion, encoder_generator,
          y_encoder_generator=y_encoder_generator, pos_encoder_generator=pos_encoder_generator,
          **args.__dict__)

