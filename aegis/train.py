"""Training: non-private training (AdamW) and person-level DP-SGD (Opacus, Poisson sampling, RDP accountant).

Methods: Training.

Non-private (PlainTraining): AdamW (betas 0.9, 0.99; weight decay on matrices), linear warm-up then cosine decay to
a tenth of the learning rate, gradient-norm clipping at 1, early stopping on the validation split; the weights of
the best epoch are kept.

Private (DPSGDTraining): DP-SGD with Opacus at the person level, one disease history being one example. Every
history enters a step independently with probability q (Poisson sampling, q = 1 / number of batches of the expected
size), each person's gradient of the per-person loss is clipped to norm C, Gaussian noise N(0, sigma^2 C^2) is added
to the sum, and the result is divided by the expected batch size qN (equation (3)). The RDP accountant sets sigma
before training so that the privacy loss after the last step equals the target epsilon at delta. There is no early
stopping and no dropout, so the released model is the one after the last accounted step. The checkpoint stores the
full privacy record.
"""
import math
import time

import numpy as np
import torch

from .data import iterate, log, make_batch, Packed
from .model import AegisModel, Config, PRESETS, save_model

# smallest fall of the validation loss that counts as an improvement for early stopping
MIN_IMPROVEMENT = 1e-4


def new_model(vocab, cfg, private, device):
    """A freshly initialised AEGIS model for the settings `cfg`.

    The size comes from PRESETS[cfg['arch']]; the hierarchy ids come from the public vocabulary, never from data.
    A private model gets no dropout, as in the paper's DP-SGD runs.

    Parameters
    ----------
    vocab : data.Vocab
    cfg : dict
        Training settings (see train()).
    private : bool
    device : str or torch.device

    Returns
    -------
    AegisModel
    """
    loss_unit = cfg.get('loss_unit', 'person')
    init_rate = cfg.get('init_rate', 'public')
    extra = dict(loss_unit=loss_unit, init_rate=init_rate)
    if cfg.get('hier', False):
        block_ids, chapter_ids = vocab.hierarchy()
        extra.update(hier_emb=True, block_ids=block_ids, chapter_ids=chapter_ids)
    dropout = 0.0 if private else cfg['dropout']
    sizes = PRESETS[cfg['arch']]
    c = Config(vocab_size=vocab.size, dropout=dropout, tie_weights=cfg['tie'], **extra, **sizes)
    model = AegisModel(c)
    return model.to(device)


@torch.no_grad()
def mean_loss(model, pk, batch, seed=7):
    """Mean per-person loss over a cohort, the validation loss.

    People without targets are left out of the mean, because their loss is 0 by definition. The batches use a
    fixed seed so that the validation loss of two epochs, or of two models, is measured on the same no-event
    tokens.

    Parameters
    ----------
    model : AegisModel
    pk : data.Packed
    batch : int
    seed : int, default 7

    Returns
    -------
    float
    """
    model.eval()
    total, n = 0.0, 0
    for _, (x, a, y, b) in iterate(pk, np.arange(pk.n), batch, seed):
        loss, n_targets = model.person_loss(x, a, y, b)
        has_targets = n_targets > 0
        batch_total = (loss * has_targets).sum()
        total += float(batch_total)
        n += int(has_targets.sum())
    return total / max(n, 1)


class PlainTraining:
    """Non-private training: AdamW, warm-up and cosine schedule, early stopping on the validation loss."""
    name = 'non-private'

    def __init__(self, cfg, device, label):
        self.cfg, self.device, self.label = cfg, device, label

    def lr_at(self, it, warm, total):
        """Learning rate at iteration `it`: linear warm-up, then cosine decay to a tenth."""
        lr = self.cfg['lr']
        if it < warm:
            return lr * (it + 1) / warm
        r = (it - warm) / max(1, total - warm)
        progress = min(r, 1)
        floor = lr / 10
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return floor + cosine * (lr - floor)

    def _optimizer(self, model):
        """AdamW with weight decay on the matrices only; biases, norms and the age map's vectors are not decayed."""
        named = list(model.named_parameters())
        decay = [p for n, p in named if p.dim() >= 2]
        nodecay = [p for n, p in named if p.dim() < 2]
        groups = [{'params': decay, 'weight_decay': self.cfg['weight_decay']},
                  {'params': nodecay, 'weight_decay': 0.0}]
        return torch.optim.AdamW(groups, lr=self.cfg['lr'], betas=(0.9, 0.99))

    def _epoch(self, model, opt, pk, gen, perm, it, warm, total):
        """One pass over the people in the order `perm`; returns the last batch loss and the iteration count."""
        cfg = self.cfg
        for s in range(0, pk.n, cfg['batch']):
            people = perm[s:s + cfg['batch']]
            idx = torch.as_tensor(people, device=self.device)
            x, a, y, b = make_batch(pk, idx, gen, train=True)
            lr = self.lr_at(it, warm, total)
            for group in opt.param_groups:
                group['lr'] = lr
            loss = model.loss(x, a, y, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            it += 1
        return loss, it

    def fit(self, model, pk, pv, gen, seed):
        """Train `model` on pk, stop early on pv, and return the weights of the best epoch.

        Parameters
        ----------
        model : AegisModel
        pk, pv : data.Packed
            Training and validation histories.
        gen : torch.Generator
            Draws the no-event tokens and lifestyle shifts of the batches.
        seed : int
            Seeds the order of the people in each epoch.

        Returns
        -------
        model, history, privacy
            The trained model, one dict per epoch, and None (no privacy record).
        """
        cfg, label = self.cfg, self.label
        history = []
        opt = self._optimizer(model)
        batches_per_epoch = math.ceil(pk.n / cfg['batch'])
        total = batches_per_epoch * cfg['epochs']
        # the warm-up never takes more than a tenth of training, so that a short run still decays
        warm = min(cfg['warmup'], total // 10 + 1)
        rng = np.random.default_rng(seed)
        best = float('inf')
        best_state = None
        # epochs since the last improvement
        bad = 0
        it = 0
        for ep in range(cfg['epochs']):
            model.train()
            perm = rng.permutation(pk.n)
            loss, it = self._epoch(model, opt, pk, gen, perm, it, warm, total)
            vl = mean_loss(model, pv, cfg['batch'] * 4)
            train_last = loss.item()
            history.append(dict(epoch=ep + 1, train_last=train_last, val=vl))
            log('  %s epoch %d: train %.3f, validation %.3f' % (label, ep + 1, train_last, vl))
            improved = vl < best - MIN_IMPROVEMENT
            # we keep a copy of the best weights, so that stopping late costs nothing but time
            if improved:
                best = vl
                bad = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad = bad + 1
            if not improved and bad >= cfg['patience']:
                log('  %s: early stop (no improvement for %d epochs)' % (label, cfg['patience']))
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history, None


class DPSGDTraining:
    """Person-level DP-SGD: Opacus, Poisson sampling, per-person clipping and the RDP accountant."""
    name = 'DP-SGD'

    def __init__(self, cfg, device, label, epsilon):
        self.cfg, self.device, self.label, self.epsilon = cfg, device, label, epsilon

    def epoch(self, model, opt, loader, pk, gen):
        """One pass over the Poisson batches, each split into physical batches that fit in memory.

        Opacus accumulates the clipped per-person gradients of the physical batches and adds the noise once per
        logical (Poisson) batch, so the privacy accounting is that of the logical batches.

        Returns
        -------
        loss : torch.Tensor
            Loss of the last physical batch.
        steps : int
            Number of logical steps taken.
        """
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        steps = 0
        with BatchMemoryManager(data_loader=loader, max_physical_batch_size=self.cfg['phys_batch'],
                                optimizer=opt) as memory_safe_loader:
            for (idx,) in memory_safe_loader:
                # Poisson sampling can draw an empty batch. Skipping it would leave the step out of the accountant,
                # which counts a step only in opt.step(), and would leave a stale flag in BatchMemoryManager's
                # queue, so that later steps mix two Poisson batches. We stop instead: the chance of an empty
                # batch is (1 - q)^N, about e^-4000 in the paper and below e^-250 in the demo and the notebook.
                if idx.numel() == 0:
                    raise RuntimeError('DP-SGD drew an empty Poisson batch; use a larger expected batch')
                idx = idx.to(self.device)
                x, a, y, b = make_batch(pk, idx, gen, train=True)
                per_person = model(x, a, b, y)
                loss = per_person.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                # one logical step per Poisson batch, not per physical batch
                steps += int(not opt._is_last_step_skipped)
        return loss, steps

    @staticmethod
    def release(model, core):
        """The plain model without Opacus' hooks and per-person gradient buffers."""
        model.remove_hooks()
        for p in core.parameters():
            p.grad = None
            for attr in ('grad_sample', 'summed_grad', '_forward_counter'):
                if hasattr(p, attr):
                    setattr(p, attr, None)
        return core

    def _make_private(self, model, pk):
        """Wrap model, optimizer and loader with Opacus; the accountant sets the noise for the target epsilon."""
        from opacus import PrivacyEngine
        from torch.utils.data import DataLoader, TensorDataset
        cfg = self.cfg
        people = TensorDataset(torch.arange(pk.n))
        expected_batch = min(cfg['dp_batch'], pk.n)
        # the sampling rate q = expected_batch / N must lie in (0, 1]; an empty training set would make it 0
        assert 0 < expected_batch <= pk.n, 'expected batch %d for %d people' % (expected_batch, pk.n)
        loader = DataLoader(people, batch_size=expected_batch, shuffle=True)
        # no weight decay in the paper's DP-SGD runs
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['dp_lr'], betas=(0.9, 0.99), weight_decay=0.0)
        # TODO: pass secure_mode=True (needs torchcsprng) when training a model meant for release; the paper's
        # audit models used the default generator, as most research use does
        engine = PrivacyEngine(accountant='rdp')
        model, opt, loader = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=loader, target_epsilon=self.epsilon, target_delta=cfg['delta'],
            epochs=cfg['dp_epochs'], max_grad_norm=cfg['clip'], poisson_sampling=True)
        return model, opt, loader, engine

    def fit(self, model, pk, pv, gen, seed):
        """Train `model` on pk with person-level DP-SGD for cfg['dp_epochs'] epochs.

        The validation loss is reported after each epoch but never used to choose the model, so the released model
        is the last one and its privacy loss is exactly the accounted one.

        Returns
        -------
        model, history, privacy
            The plain trained model, one dict per epoch, and the privacy record (epsilon spent, delta, noise
            multiplier, sampling rate, steps, clip).
        """
        cfg, label, epsilon = self.cfg, self.label, self.epsilon
        history = []
        model, opt, loader, engine = self._make_private(model, pk)
        core = model._module
        q = loader.sample_rate
        log('  %s: epsilon %.2f, delta %.0e, noise multiplier %.3f, sampling rate %.5f (expected batch %.0f), '
            '%d epochs, clip %.2f' % (label, epsilon, cfg['delta'], opt.noise_multiplier, q, q * pk.n,
                                     cfg['dp_epochs'], cfg['clip']))
        steps = 0
        for ep in range(cfg['dp_epochs']):
            model.train()
            loss, n = self.epoch(model, opt, loader, pk, gen)
            steps += n
            eps_spent = engine.get_epsilon(cfg['delta'])
            vl = mean_loss(core, pv, 1024)
            train_last = loss.item()
            history.append(dict(epoch=ep + 1, train_last=train_last, val=vl, epsilon=eps_spent))
            log('  %s epoch %d: train %.3f, validation %.3f, epsilon spent %.2f' % (label, ep + 1, train_last, vl,
                                                                                  eps_spent))
        privacy = dict(target_epsilon=epsilon, epsilon=float(engine.get_epsilon(cfg['delta'])), delta=cfg['delta'],
                       noise_multiplier=float(opt.noise_multiplier), sample_rate=float(q),
                       expected_batch=float(q * pk.n), epochs=cfg['dp_epochs'], steps=steps, clip=cfg['clip'],
                       accountant='rdp', n_train=pk.n, unit='person (one history per participant)')
        return self.release(model, core), history, privacy


def train(train_coh, val_coh, vocab, cfg, device, epsilon=None, seed=0, label='model', out_path=None):
    """Train one AEGIS model, without privacy or with person-level DP-SGD at a target epsilon.

    The seed fixes the initial weights, the order of the people and every random draw of the batches, so two calls
    with the same arguments give the same model on the same device.

    Parameters
    ----------
    train_coh, val_coh : data.Cohort
        Training and validation histories.
    vocab : data.Vocab
    cfg : dict
        arch, max_events, epochs, batch, lr, weight_decay, warmup, patience, dropout, tie, hier, loss_unit,
        init_rate; for DP-SGD also dp_epochs, dp_batch (expected batch), phys_batch (people per pass), dp_lr,
        clip and delta.
    device : str or torch.device
    epsilon : float, optional
        Target epsilon; None or 0 trains without privacy.
    seed : int, default 0
    label : str
        Name used in the log.
    out_path : str, optional
        Where to save the checkpoint.

    Returns
    -------
    model : AegisModel
        In eval mode.
    extra : dict
        label, private, privacy record, history per epoch, seed, minutes, n_train and the settings.
    """
    torch.manual_seed(seed)
    t0 = time.time()
    pk = Packed(train_coh, cfg['max_events'], device)
    pv = Packed(val_coh, cfg['max_events'], device)
    private = epsilon is not None and epsilon > 0
    strategy = DPSGDTraining(cfg, device, label, epsilon) if private else PlainTraining(cfg, device, label)
    model = new_model(vocab, cfg, private, device)
    log('  %s: %s, %.2fM parameters, %d training histories'
        % (label, strategy.name, model.n_params() / 1e6, pk.n))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model, history, privacy = strategy.fit(model, pk, pv, gen, seed)
    model.eval()
    extra = dict(label=label, private=private, privacy=privacy, history=history, seed=seed,
                 minutes=(time.time() - t0) / 60, n_train=pk.n, train_cfg={k: cfg[k] for k in sorted(cfg)})
    if out_path:
        save_model(model, out_path, extra)
    log('  %s: done in %.1f min' % (label, extra['minutes']))
    return model, extra
