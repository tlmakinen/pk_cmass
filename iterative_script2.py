#!/usr/bin/env python
# ----------------------------------------------
# train_hybrid_pk.py
# ----------------------------------------------


from new_epe_code import *          # noqa: F403, provides EPEModel, MDN, etc.

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# --- ILI / local imports ------------------------------------------------------
import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.embedding import FCN


# -----------------------------------------------------------------------------


# ----------------------- DATA -------------------------------------------------
def load_data(folder: Path,
              n_params: int,
              pk_cut: int = 94) -> Tuple[np.ndarray, ...]:
    """
    Load raw data arrays and split into pk / bk helpers.
    """
    x_train = np.load(folder / "x_train.npy")
    theta_train = np.load(folder / "theta_train.npy")[:, :n_params]

    x_test = np.load(folder / "x_test.npy")
    theta_test = np.load(folder / "theta_test.npy")[:, :n_params]

    x_val = np.load(folder / "x_val.npy")
    theta_val = np.load(folder / "theta_val.npy")[:, :n_params]

    return (x_train, theta_train,
            x_val, theta_val,
            x_test, theta_test)


def build_standardisers(x_train: np.ndarray):
    mean = x_train.mean(0)
    std = np.sqrt((x_train ** 2).mean(0) - mean ** 2)

    def get_pk(data, cut):
        data = (data - mean) / std
        return jnp.concatenate([data[:cut], data[-10:]]) # concatenate nbar

    def get_bk(data, start, stop):
        data = (data - mean) / std
        return data[start:stop]

    return get_pk, get_bk, mean, std


@jax.jit
def optimized_smooth_leaky(x: jnp.ndarray, alpha: float = 0.2) -> jnp.ndarray:
    """
    A C-infinity smooth version of a leaky ReLU.
    Combines linear growth with a smooth transition at the origin.
    """
    # Using a softplus-based leak ensures no 'kinks' at -1 or 1
    return alpha * x + (1 - alpha) * jax.nn.log_sigmoid(x) * x


def mish(x):
    """
    Mish: A Self-Regularized Non-Monotonic Activation Function.
    Formula: x * tanh(softplus(x))
    """
    return x * jnp.tanh(jax.nn.softplus(x))

# BASE RESIDUAL MLP MODEL
class ResMLP(nn.Module):
    features: Sequence[int]
    act: Callable = nn.relu
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        x = self.act(nn.Dense(self.features[0])(x))
        for feat in self.features[1:-1]:
            x1 = nn.Dense(feat)(x)
            x += x1
            x = self.act(x)
        x = nn.Dense(self.features[-1])(x)
        if self.activate_final:
            x = self.act(x)
        return x


# --------------------- MODEL HELPERS -----------------------------------------
def init_models(experiment: str,
                n_params: int,
                n_summs_pk: int,
                n_summs_bk: int,
                mdn_components: int,
                # n_summs_hybrid: int,
                theta_train: np.ndarray,
                inds: Tuple,
                pk_compress: Callable = None):


    theta_fid = theta_train.mean(0)



    class CompressPk(EPEModel, nn.Module):
        n_summaries: int
        cut: int
        n_params: int = 3
        n_components: int = 4

        def setup(self):
            self.mdn = MDN(hidden_channels=[128],
                           n_components=self.n_components,
                           n_dimension=self.n_params,
                           theta_star=jnp.array(theta_fid))
            self.mlp = ResMLP(features=[500, 500, 500, self.n_summaries],
                              act=smooth_leaky)
            self.norm = nn.LayerNorm()

        def get_embed(self, x):
            x = get_pk(x, self.cut)
            print("pk cut", x.shape)
            x = self.mlp(x)
            return self.norm(x)

        def log_prob(self, x, theta):
            x = self.get_embed(x)
            return self.mdn(x, theta - theta_fid)

        __call__ = log_prob


    class CompressBk(EPEModel, nn.Module):
        n_summaries: int
        n_params: int = 3
        n_components: int = 4
        inds: Tuple = (0, 73)

        def setup(self):
            self.mdn = MDN(hidden_channels=[128],
                           n_components=self.n_components,
                           n_dimension=self.n_params,
                           theta_star=jnp.array(theta_fid))
            self.mlp = ResMLP(features=[500, 500, 500, self.n_summaries],
                              act=smooth_leaky)
            self.norm = nn.LayerNorm()

        def get_embed(self, x):
            bk = self.mlp(get_bk(x, self.inds[0], self.inds[1]))
            return self.norm(bk)

        def log_prob(self, x, theta):
            x = self.get_embed(x)
            return self.mdn(x, theta - theta_fid)

        __call__ = log_prob

    class HybridNet(EPEModel, nn.Module):
        n_summaries: int
        pk_net: Callable
        n_params: int = 3
        n_components: int = 4
        #inds: list[Tuple] = [(0,73), (73,307), (307,415)]
        inds: Tuple = (0, 73)
        act: Callable = smooth_leaky
        # pca_layer: nn.Module = None

        def setup(self):
            self.mdn = MDN(hidden_channels=[128],
                           n_components=self.n_components,
                           n_dimension=self.n_params,
                           theta_star=jnp.array(theta_fid))
            self.mdn2 = MDN(hidden_channels=[128],
                           n_components=self.n_components,
                           n_dimension=self.n_params,
                           theta_star=jnp.array(theta_fid))
            self.mlp = ResMLP(features=[500, 500, 500, self.n_summaries],
                              act=self.act)
            self.norm = nn.LayerNorm()
            self.norm2 = nn.LayerNorm()
            

        def get_embed(self, x, existing=False):
            # if self.pca_layer is not None:
            #     res = self.pca_layer(x)

            pk = self.pk_net(x)
            print("pk", pk.shape)
            bk = self.mlp(get_bk(x, self.inds[0], self.inds[1]))

            if existing:
                return self.norm(jnp.concatenate([pk, bk], -1)), self.norm2(pk)
            else:
                return self.norm(jnp.concatenate([pk, bk], -1))

        def log_prob(self, x, theta):
            x, existing = self.get_embed(x, existing=True)
            dummy = x #jnp.ones_like(x)
            dummy = dummy.at[:existing.shape[0]].set(0.0)
            return self.mdn(x, theta - theta_fid) #+ self.mdn(dummy, theta)

        __call__ = log_prob


    model_pk = CompressPk(n_summaries=n_summs_pk, n_params=n_params, n_components=mdn_components, cut=inds[0])
    model_bk = CompressBk(n_summaries=n_summs_bk, n_params=n_params, n_components=mdn_components, inds=inds)
    pk_net = pk_compress if pk_compress is not None else lambda x: x
    model_hybrid = HybridNet(n_summaries=n_summs_bk, n_params=n_params, n_components=mdn_components, inds=inds, pk_net=pk_compress)

    if experiment == "hybrid":
        return model_pk, model_hybrid

    elif experiment == "pk_bk_separate":
        return model_pk, model_bk

    else:
        return model_pk
    # different options        


# --------------------- TRAINING LOOP MAIN SCRIPT -----------------------------------------


def _run_embedding_loop(model, 
                             key,
                             train_data, # tuple of (d ,theta)
                             test_data,
                             epochs=1000,
                             batch_size=64,
                             learning_rate=5e-6,
                            n_params=5,
                            w=None):

    
    data_train, theta_train = train_data
    data_test, theta_test =  test_data

    data_single_shape = data_train[0].shape
    
    n_train = data_train.shape[0]
    
    remainder = batch_size * (data_train.shape[0] // batch_size)
    
    data_ = data_train[:remainder].reshape((-1, batch_size,) + data_single_shape)
    theta_ = theta_train[:remainder].reshape(-1, batch_size, n_params)
    
    # reshape the test data into batches
    remainder = batch_size * (data_test.shape[0] // batch_size)

    data_test = data_test[:remainder].reshape((-1, batch_size,) + data_single_shape)
    theta_test = theta_test[:remainder].reshape(-1, batch_size, n_params)

    @jax.jit
    def logprob_loss(w, x_batched, theta_batched):
    
        def fn(x, theta):
           logp = model.apply(w, x, theta)
           return logp
    
        logp_batched = jax.vmap(fn)(x_batched, theta_batched)
        return -jnp.mean(logp_batched)

    
    # init model again
    if w is None:
        w = model.init(key, data_train[0], jnp.ones(n_params,))
    
    # # Clip gradients at max value, and evt. apply weight decay
    transf = [optax.clip(2.0)]
    transf.append(optax.add_decayed_weights(1e-4))
    tx = optax.chain(
        *transf,
        optax.adam(learning_rate=learning_rate)
    )
    opt_state = tx.init(w)
    loss_grad_fn = jax.value_and_grad(logprob_loss)

    
    # this is a hack to make the for-loop training much faster in jax
    def body_fun(i, inputs):
        w,loss_val, opt_state, _data, _theta, key = inputs
        x_samples = _data[i]
        y_samples = _theta[i]
    
        # apply noise simulator
        keys = jr.split(key, x_samples.shape[0])
        #x_samples = jax.vmap(noise_simulator)(keys, x_samples)
    
    
        loss_val, grads = loss_grad_fn(w, x_samples, y_samples)
        updates, opt_state = tx.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
    
        return w, loss_val, opt_state, _data, _theta, key
    
    
    def val_body_fun(i, inputs):
        w,loss_val, _data, _theta, key = inputs
        x_samples = _data[i]
        y_samples = _theta[i]
    
        # apply noise simulator
        keys = jr.split(key, x_samples.shape[0])
        #x_samples = jax.vmap(noise_simulator)(keys, x_samples)
    
        loss_val, grads = loss_grad_fn(w, x_samples, y_samples)
    
        return w, loss_val, _data, _theta, key
    

    
    losses = jnp.zeros(epochs)
    val_losses = jnp.zeros(epochs)
    loss_val = 0.
    val_loss_value = 0.
    best_val_loss = jnp.inf
    lower = 0
    upper = n_train // batch_size

    best_w = w
    
    pbar = tqdm(range(epochs), leave=True, position=0)
    counter = 0
    
    for j in pbar:
          key,rng = jax.random.split(key)
    
          # shuffle data every epoch
          randidx = jr.permutation(key, jnp.arange(theta_.reshape(-1, n_params).shape[0]), independent=True)
          _data = data_.reshape((-1,) + data_single_shape)[randidx].reshape((-1, batch_size,) + data_single_shape)
          _theta = theta_.reshape(-1, n_params)[randidx].reshape(-1, batch_size, n_params)
    
          #print(_data.shape)
    
          inits = (w, loss_val, opt_state, _data, _theta, key)
          w, loss_val, opt_state, _data, _theta, key = jax.lax.fori_loop(lower, upper, body_fun, inits)
          losses = losses.at[j].set(loss_val)
    
    
          # do validation set
          key,rng = jr.split(key)
          inits = (w, loss_val, data_test, theta_test, key)
          w, val_loss_value, data_test, theta_test, key = jax.lax.fori_loop(0, data_test.shape[0], val_body_fun, inits)
          val_losses = val_losses.at[j].set(val_loss_value)
    
          #val_losses.append(val_loss)
          pbar.set_description('epoch %d loss: %.5f  val loss: %.5f'%(j, loss_val, val_loss_value))


          if val_loss_value < best_val_loss:
              best_val_loss = val_loss_value
              best_w = w
    
    
          counter += 1

    return best_w, (losses, val_losses)



# ------------------ TRAINING --------------------------------------------------
def run_embedding_loop(model, key, train_data, val_data,
                       epochs=250, batch_size=64, lr=5e-6, n_params=5):
    """
    Thin wrapper around the original training loop.
    """
    # the original `run_embedding_loop` from the notebook can be imported
    # directly.  Here we simply re-expose it with cleaner defaults.
    return _run_embedding_loop(model, key, train_data, val_data,
                               epochs=epochs, batch_size=batch_size,
                               learning_rate=lr, n_params=n_params) 


# --------------- POST-TRAINING HELPERS ---------------------------------------
def summaries_from_model(model, weights, data):
    vmap_apply = jax.vmap(lambda d: model.apply(weights, d, method=model.get_embed))
    return np.array(vmap_apply(data))


# --------------- MAIN ---------------------------------------------------------
def main(args):
    # device selection ------------------------- --------------------------------
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    # if device == 'cupu':
        # jax.config.update('jax_default_device', jax.devices('cpu')[0])

    print(f"Device: {device}")

    folder = Path(args.folder)
    test_folder = Path(args.test_folder)
    n_params = 5

    # 1. data ------------------------------------------------------------------
    (x_tr, th_tr,
     x_val, th_val,
     x_te, th_te) = load_data(folder, n_params)

    # pull in test data if different one specified
    (_, _,
     _, _,
     x_te, th_te) = load_data(test_folder, n_params)

    # 2. preprocessing ---------------------------------------------------------
    global get_pk, get_bk        # share with model definition
    get_pk, get_bk, mean_cl, std_cl = build_standardisers(x_tr)

    # 3. models ----------------------------------------------------------------

    # take a look at what experiment we're dealing with
    # if experiment == "hybrid":
    #     return model_pk, model_hybrid

    # elif experiment == "pk_bk_separate":
    #     return model_pk, model_bk

    # else:
    #     return model_pk
    # different options     

    experiment = args.experiment

    global pk_net

    if experiment == "hybrid":

        from copy import deepcopy

        # if args.noise_test:
        #     print('testing dummy noise placeholders')
        #     experiment = "hybrid_noise"


        def fit_pca(data, n_components):
            # Center the data
            mean = jnp.mean(data, axis=0)
            centered_data = data - mean
            # Compute covariance and eigen-decomposition
            cov = jnp.cov(centered_data, rowvar=False)
            evals, evecs = jnp.linalg.eigh(cov)
            # Sort descending and slice
            idx = jnp.argsort(evals)[::-1][:n_components]
            return mean, evecs[:, idx]

        mean_pca, w_pca = fit_pca(x_tr, n_components=args.pk_summaries + args.bk_summaries)

        
        # indexes = [(73,307), (307,415)]
        indexes = [(63,126), (126,188), (188,307), (307,415)] # cumulative indexes

        # Validate compression stage argument
        if args.compression_stage < 0 or args.compression_stage >= len(indexes):
            raise ValueError(f"compression_stage must be between 0 and {len(indexes)-1}, got {args.compression_stage}")
    
        key = jr.PRNGKey(args.seed)

        # Initial iteration: train PK compressor on indexes[0]
        print(f"Training PK compressor on indexes[0]={indexes[0]} …")
        model_pk, _ = init_models(experiment,
                                  n_params,
                                  n_summs_pk=args.pk_summaries,
                                  n_summs_bk=args.bk_summaries,
                                  mdn_components=args.mdn_components,
                                  theta_train=th_tr,
                                  inds=indexes[0])
    
        w_pk, _ = run_embedding_loop(model_pk, key,
                                     (x_tr, th_tr), (x_val, th_val),
                                     epochs=args.epochs, n_params=n_params)
    
        pk_net = lambda d: model_pk.apply(w_pk, d, method=model_pk.get_embed)

        # First hybrid stage on indexes[0]
        print(f"Training hybrid compressor on stage 0 (indexes[0]={indexes[0]}) …")
        _, model_hybrid = init_models(experiment,
                                      n_params,
                                      n_summs_pk=args.pk_summaries,
                                      n_summs_bk=args.bk_summaries,
                                      mdn_components=args.mdn_components,
                                      theta_train=th_tr,
                                      inds=indexes[0], 
                                      pk_compress=pk_net)
    
        w_hybrid, _ = run_embedding_loop(model_hybrid, key,
                                        (x_tr, th_tr), (x_val, th_val),
                                        epochs=args.epochs, n_params=n_params)

        # Cumulative compression: iterate through remaining stages up to compression_stage
        for stage_idx in range(1, args.compression_stage + 1):
            print(f"Training hybrid compressor on stage {stage_idx} (indexes[{stage_idx}]={indexes[stage_idx]}) …")
            
            # Previous hybrid becomes new pk_net
            pk_net = lambda d, w=w_hybrid, m=model_hybrid: m.apply(w, d, method=m.get_embed)

            _, model_hybrid = init_models(experiment,
                                          n_params,
                                          n_summs_pk=args.pk_summaries,
                                          n_summs_bk=args.bk_summaries,
                                          mdn_components=args.mdn_components,
                                          theta_train=th_tr,
                                          inds=indexes[stage_idx], 
                                          pk_compress=pk_net)

            w_hybrid, _ = run_embedding_loop(model_hybrid, key,
                                            (x_tr, th_tr), (x_val, th_val),
                                            epochs=args.epochs, n_params=n_params)

        # obtain summaries from the final trained model
        print(f"Generating summaries from compression stage {args.compression_stage} …")
        summ_train = summaries_from_model(model_hybrid, w_hybrid, x_tr)
        summ_val   = summaries_from_model(model_hybrid, w_hybrid, x_val)
        summ_test  = summaries_from_model(model_hybrid, w_hybrid, x_te)

    elif experiment == "pk_bk_separate":

        key = jr.PRNGKey(args.seed)
        model_pk, model_bk = init_models(experiment,
                                             n_params,
                                             n_summs_pk=args.pk_summaries,
                                             n_summs_bk=args.bk_summaries,
                                             mdn_components=args.mdn_components,
                                             theta_train=th_tr)
    
        # train PK compressor ------------------------------------------------------
        print("Training Pk compressor …")
        w_pk, _ = run_embedding_loop(model_pk, key,
                                     (x_tr, th_tr), (x_val, th_val),
                                     epochs=args.epochs, n_params=n_params)
    
        # global pk_net
        pk_net = lambda d: model_pk.apply(w_pk, d, method=model_pk.get_embed)
    
        # train hybrid -------------------------------------------------------------
        print("Training Bk compressor …")
        w_bk, _ = run_embedding_loop(model_bk, key,
                                         (x_tr, th_tr), (x_val, th_val),
                                         epochs=args.epochs, n_params=n_params)
    
        # obtain summaries ---------------------------------------------------------
        summ_train = summaries_from_model(model_pk, w_pk, x_tr)
        summ_val   = summaries_from_model(model_pk, w_pk, x_val)
        summ_test  = summaries_from_model(model_pk, w_pk, x_te)

        summ_train_bk = summaries_from_model(model_bk, w_bk, x_tr)
        summ_val_bk   = summaries_from_model(model_bk, w_bk, x_val)
        summ_test_bk  = summaries_from_model(model_bk, w_bk, x_te)

        summ_train = np.concatenate([summ_train, summ_train_bk], -1)
        summ_val = np.concatenate([summ_val, summ_val_bk], -1)
        summ_test = np.concatenate([summ_test, summ_test_bk], -1)


    elif experiment == "pk_only_compression":
        key = jr.PRNGKey(args.seed)
        model_pk = init_models(experiment,
                                             n_params,
                                             n_summs_pk=args.pk_summaries,
                                             n_summs_bk=args.bk_summaries,
                                             mdn_components=args.mdn_components,
                                             theta_train=th_tr)
    
        # train PK compressor ------------------------------------------------------
        print("Training PK compressor …")
        w_pk, _ = run_embedding_loop(model_pk, key,
                                     (x_tr, th_tr), (x_val, th_val),
                                     epochs=args.epochs, n_params=n_params)
    
        # global pk_net
        pk_net = lambda d: model_pk.apply(w_pk, d, method=model_pk.get_embed)
    
        # obtain summaries ---------------------------------------------------------
        summ_train = summaries_from_model(model_pk, w_pk, x_tr)
        summ_val   = summaries_from_model(model_pk, w_pk, x_val)
        summ_test  = summaries_from_model(model_pk, w_pk, x_te)

    

    # HERE WE PROCEED WITH NO COMPRESSION
    elif experiment == "pk_bk_nocompress":
        print("proceeding without compression of pk+bk")
        summ_train = x_tr
        summ_val = x_val
        summ_test = x_te


    elif experiment == "pk_only_nocompress":
        print("proceeding without compression of pk only")
        summ_train = jax.vmap(get_pk)(x_tr)
        summ_val = jax.vmap(get_pk)(x_val)
        summ_test = jax.vmap(get_pk)(x_te)

        print("summ train", summ_train.shape)


    elif experiment == "bk_only_nocompress":
        print("proceeding without compression of bk only")
        summ_train = jax.vmap(get_bk)(x_tr)
        summ_val = jax.vmap(get_bk)(x_val)
        summ_test = jax.vmap(get_bk)(x_te)


    # Specify your directory name
    dir_path = Path(args.outfile)

    # Create the directory
    dir_path.mkdir(parents=True, exist_ok=True)
    
    
    filename = f"summaries_{experiment}.npz"
    outfname = dir_path / filename

    np.savez(outfname,
             summs_train=summ_train, theta_train=th_tr,
             summs_val=summ_val,   theta_val=th_val,
             summs_test=summ_test, theta_test=th_te,
             n_summs_pk=args.pk_summaries,
             n_summs_bk=args.bk_summaries)

    # -------------------------------------------------------------------------
    print(f"Saved compressed summaries to {outfname}")

    
    


    # next run the ILI loop on the chosen summaries

    from ili.embedding import FCN

    # set common manual seed
    torch.manual_seed(args.seed)

    outdir = args.outfile + "npe_%s"%(experiment)
    
    activation = "LeakyReLU"
    n_hidden = [128, 128, 6]
    embedding_network = FCN(n_hidden = n_hidden, act_fn = activation)
        
    
    # set train / val index
    val_start = summ_train.shape[0]
    train_idx = torch.arange(0, summ_train.shape[0])
    val_idx = torch.arange(val_start, val_start + summ_val.shape[0])
    
    loader = NumpyLoader(x=np.concatenate([summ_train, summ_val],0 ),
                                 theta=np.concatenate([th_tr, th_val], 0))
    
    # define a prior for the scaled thetas
    
    prior = ili.utils.Uniform(
        low=th_tr.min(axis=0),
        high=th_tr.max(axis=0),
        device=device)
    
    
    # instantiate your neural networks to be used as an ensemble
    nets = [
        ili.utils.load_nde_sbi(engine='NPE', model='nsf', hidden_features=100, num_transforms=12, embedding_net=embedding_network),
        ili.utils.load_nde_sbi(engine='NPE', model='nsf', hidden_features=100, num_transforms=7, embedding_net=embedding_network),
        ili.utils.load_nde_sbi(engine='NPE', model='maf', hidden_features=100, num_transforms=9, embedding_net=embedding_network),
        ili.utils.load_nde_sbi(engine='NPE', model='maf', hidden_features=100, num_transforms=12, embedding_net=embedding_network),
        # ili.utils.load_nde_sbi(engine='NPE', model='mdn', hidden_features=100, num_components=9, embedding_net=embedding_network),
    ]
    
    # define training arguments
    train_args = {
        'training_batch_size': 64,
        'learning_rate': 1e-5,
        'validation_fraction': 0.25,
        'stop_after_epochs': 10
        
    }
    
    # initialize the trainer
    runner = InferenceRunner.load(
        backend='sbi',
        engine='NPE',
        prior=prior,
        nets=nets,
        device=device,
        #embedding_net=embedding_network,
        train_args=train_args,
        proposal=None,
        out_dir=outdir,            
        train_indices=train_idx,
        val_indices=val_idx
    )
    
    posterior_ensemble, summaries = runner(loader=loader)
    
    # ----------------------------------------
    
    
    
    # look at a bunch of predictions around a fiducial cosmology
    theta_pl18 = np.array([0.3158, 0.04897, 0.67, 0.96,  0.8120])
    # get a set of simulations within a small ball
    eps = 0.085
        
    dists = np.linalg.norm((th_te - theta_pl18), axis=-1)
    mask = dists < eps
    print(mask.sum())

    # convert back to numpy
    summ_test = np.array(summ_test)
    
    all_fid_chains = np.array([posterior_ensemble.sample(sample_shape=(10000,), x=torch.tensor(s).to(device)).cpu().numpy() for s in summ_test[mask]])
    
    print(f"saved fiducial inference chains to {args.outfile}")

    outchains = args.outfile + "fiducial_chains_%s"%(experiment)
    np.savez(outchains,
             chains=all_fid_chains,
           )

    # run validation tests
    print('running validation checks on Quijote')

    plotdir = args.outfile + "quijote_plots_%s"%(experiment)
    Path(plotdir).mkdir(parents=True, exist_ok=True)  # 

    
    labels = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct',
        labels=labels,
        plot_list = ["coverage", "histogram", "predictions", "tarp"],
        out_dir=plotdir
    ) 

    skip = 2

    fig = metric(
        posterior=posterior_ensemble, # NeuralPosteriorEnsemble instance from sbi package
        x=summ_test[::skip], theta=th_te[::skip]
    )
    plt.clf() #release memory
    plt.close() #kill the figure    


    



# ------------------ CLI -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PK/Hybrid compressors and save summaries."
    )
    parser.add_argument("--folder", required=True,
                        help="Folder containing x_train.npy etc.")

    parser.add_argument("--test-folder", required=True,
                        help="Folder containing the suite to test on.")

    parser.add_argument("--experiment", type=str, default="hybrid")
    
    parser.add_argument("--epochs",  type=int, default=250)
    parser.add_argument("--pk-summaries",     type=int, default=7)
    parser.add_argument("--pk-cut", type=int, default=94)
    parser.add_argument("--mdn-components", type=int, default=7)
    parser.add_argument("--bk-summaries", type=int, default=7)
    parser.add_argument("--compression-stage", type=int, default=1,
                        help="Index of compression stage to test (0-based). Will cumulatively compress from indexes[0] to indexes[compression_stage].")
    parser.add_argument("--outfile", default="/data101/makinen/pk_cmass/compare_comp_16_01/")
    parser.add_argument("--device",  default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())