"""Synthetic cohorts: sampling, the copying test, the DCR check and rate calibration.

Methods: Synthetic cohorts; Rate calibration and the independent-rates generator.

Each synthetic person starts at birth with a random sex. Sampling follows Delphi's competing exponentials: every
predictable token k gets a waiting time t_k ~ Exp(rate_k), with rate_k = exp(logit_k) per day; the next event is the
argmin and the clock advances by min t_k. A disease occurs only once; sampling stops at death or at 85 years.
Synthetic people carry no lifestyle tokens, because lifestyle is an input the model never predicts. A cohort
sampled from a DP-SGD model keeps the model's (epsilon, delta) guarantee (post-processing); a cohort from the
non-private model has no guarantee.

The copying test takes, for each planted and kept-aside patient, the largest Jaccard similarity of their code set
to any synthetic person, and the AUC of planted against kept-aside patients: a cohort that copies training records
separates them. The DCR check (distance to closest record) gives the share of 5,000 synthetic people whose Jaccard
distance to the closest of 20,000 training histories is smaller than to the closest of 20,000 non-members (ties
counted separately).

Rate calibration: DP-SGD noise moves the overall rate level of a private model from seed to seed, and a model whose
rates are too low generates synthetic people with too few diagnoses. Calibration multiplies every rate by one factor
c (adds log c to every logit), chosen so that synthetic people have the real number of disease events per
person-year between ages 40 and 70:
  * real side: two totals of the training set, disease events in the window (at most 60 per person) and
    person-years in the window (at most 30 per person), released with the Laplace mechanism at epsilon 0.005 each
    (0.01 in total, added to the model's budget by basic composition);
  * synthetic side: sampled from the model itself (post-processing, no privacy cost); c is updated a few times.
As one constant is added to every logit, rankings and AUCs are unchanged; only the waiting times change.
"""
import math
import warnings

import numpy as np
import torch
from scipy.sparse import csr_matrix

from .data import auc, Cohort, FEMALE, MALE, MASK_TIME, MIN_CELL, NO_EVENT, PAD, suppress_small
from .model import NEG

# smallest uniform draw: -log(u) must stay finite
U_FLOOR = 1e-12
# two log-scales closer than this are the same point, and no secant slope can be taken from them
SAME_SCALE = 1e-6


@torch.no_grad()
def sample_cohort(model, vocab, n, device, batch=2048, max_age_years=85.0, max_steps=160, seed=0):
    """Sample n synthetic people from birth with the model's competing exponentials.

    People are sampled in batches; a person stops at death, at max_age_years, or after max_steps records (no-event
    tokens included), a cap that bounds the run time of a model that rarely predicts death. Every random draw comes
    from one generator seeded with `seed`, so a cohort is reproducible.

    Parameters
    ----------
    model : AegisModel
    vocab : data.Vocab
    n : int
    device : str or torch.device
    batch : int, default 2048
    max_age_years : float, default 85.0
    max_steps : int, default 160
    seed : int, default 0

    Returns
    -------
    data.Cohort
        The synthetic people, with their end of follow-up (death, else the age limit).
    """
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    P, K, A, END = [], [], [], []
    max_age = max_age_years * 365.25
    for base in range(0, n, batch):
        B = min(batch, n - base)
        x, a, alive = _sample_batch(model, vocab, B, g, device, max_age, max_steps)
        xs = x.cpu().numpy()
        ages = a.cpu().numpy()
        END.append(_end_of_follow_up(xs, ages, alive, vocab, max_age))
        keep = (xs != PAD) & (xs != NO_EVENT)
        r, c = np.nonzero(keep)
        P.append(r + base)
        K.append(xs[r, c])
        A.append(ages[r, c])
    person = np.concatenate(P)
    tok = np.concatenate(K).astype(np.int32)
    age = np.concatenate(A).astype(np.float32)
    coh = Cohort.from_long(person, tok, age)
    end = np.concatenate(END).astype(np.float32)
    coh.end = end[:len(coh)]
    return coh


def _sample_batch(model, vocab, B, g, device, max_age, max_steps):
    """Sample B people from birth; returns their tokens and ages [B, steps + 1] and who is still alive."""
    draw = torch.rand(B, generator=g, device=device)
    female = torch.full((B,), FEMALE, device=device)
    male = torch.full((B,), MALE, device=device)
    sex = torch.where(draw < 0.5, female, male)
    x = sex[:, None].long()
    a = torch.zeros(B, 1, device=device)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    seen = torch.zeros(B, vocab.size, dtype=torch.bool, device=device)
    for _ in range(max_steps):
        if not alive.any():
            break
        x, a = _next_record(model, vocab, x, a, alive, seen, B, g, device, max_age)
    return x, a, alive


def _next_record(model, vocab, x, a, alive, seen, B, g, device, max_age):
    """Draw the next record of everyone still alive (updates `alive` and `seen` in place)."""
    idx = alive.nonzero(as_tuple=True)[0]
    all_logits = model.logits(x[idx], a[idx])
    lg = all_logits[:, -1].float()
    # a disease occurs only once
    lg = lg.masked_fill(seen[idx], NEG)
    u = torch.rand(lg.shape, generator=g, device=device)
    u = u.clamp_min(U_FLOOR)
    exponential = -torch.log(u)
    # waiting time of every token in days; the next event is the first to happen
    t = exponential * torch.exp(-lg)
    dt, nxt = t.min(1)
    dt = dt.clamp(max=80 * 365.25)
    new_age = a[idx, -1] + dt
    stop = new_age > max_age
    new_tok = torch.full((B,), PAD, dtype=torch.long, device=device)
    new_ages = torch.full((B,), MASK_TIME, device=device)
    ok = ~stop
    new_tok[idx[ok]] = nxt[ok]
    new_ages[idx[ok]] = new_age[ok]
    x = torch.cat([x, new_tok[:, None]], 1)
    a = torch.cat([a, new_ages[:, None]], 1)
    disease = ok & (nxt != NO_EVENT)
    seen[idx[disease], nxt[disease]] = True
    alive[idx[stop | (ok & (nxt == vocab.death))]] = False
    return x, a


def _end_of_follow_up(xs, ages, alive, vocab, max_age):
    """End of follow-up per person: the age at death, else the age limit, or the last age where sampling stopped."""
    died = (xs == vocab.death)
    anyone_died = died.any(1)
    death_age = np.where(died, ages, -np.inf).max(1)
    if alive.any():
        still_alive = alive.cpu().numpy()
        end_otherwise = np.where(still_alive, ages.max(1), max_age)
    else:
        end_otherwise = max_age
    return np.where(anyone_died, death_age, end_otherwise)


class ModelGenerator:
    """Synthetic people sampled from a trained model, one record at a time."""

    def __init__(self, model, device, batch=2048):
        self.model, self.device, self.batch = model, device, batch

    def sample(self, vocab, n, seed=0):
        return sample_cohort(self.model, vocab, n, self.device, batch=self.batch, seed=seed)


def code_matrix(coh, vocab):
    """People x tokens indicator matrix (sparse) of the codes and death each person has."""
    rows = coh.rows()
    ok = coh.tok >= vocab.first_code
    ones = np.ones(ok.sum(), np.float32)
    people = rows[ok]
    tokens = coh.tok[ok].astype(np.int64)
    M = csr_matrix((ones, (people, tokens)), shape=(len(coh), vocab.size))
    # a token recorded twice for one person would sum to 2; a code set has ones only
    M.data[:] = 1
    return M


def _best_jaccard(C, S, chunk=50000):
    """For each row of C, the highest Jaccard similarity to any row of S (binary code sets).

    We work through S in chunks of rows so that the dense intersection matrix stays small.
    """
    c_sizes = np.asarray(C.sum(1)).ravel()
    s_sizes = np.asarray(S.sum(1)).ravel()
    best = np.zeros(C.shape[0])
    for s in range(0, S.shape[0], chunk):
        block = S[s:s + chunk]
        inter = (C @ block.T).toarray()
        union = c_sizes[:, None] + s_sizes[None, s:s + chunk] - inter
        # two empty code sets have similarity 0 here, not 0 / 0
        union = np.maximum(union, 1)
        similarity = inter / union
        best = np.maximum(best, similarity.max(1))
    return best


def copying_test(planted, kept_aside, synth, vocab, near=0.75):
    """Copying test: does a synthetic cohort contain near copies of the patients planted in training?

    For each planted and kept-aside patient we take the Jaccard similarity of their code set to the closest
    synthetic person. If the model copies training records, planted patients have closer matches than kept-aside
    ones and the AUC is above 0.5. The patients are artificial, so the histograms are not suppressed.

    Parameters
    ----------
    planted : dict of str to data.Cohort
        Patients planted once ('1') and eight times ('8').
    kept_aside : data.Cohort
    synth : data.Cohort
    vocab : data.Vocab
    near : float, default 0.75
        Similarity counted as a near copy.

    Returns
    -------
    dict of str to dict
        Per planted set: AUC, mean similarities, histograms and the number of near copies.
    """
    S = code_matrix(synth, vocab)
    kept_aside_codes = code_matrix(kept_aside, vocab)
    best_out = _best_jaccard(kept_aside_codes, S)
    e = np.linspace(0, 1, 21)
    out = {}
    for k, c in planted.items():
        planted_codes = code_matrix(c, vocab)
        best_in = _best_jaccard(planted_codes, S)
        out[k] = _copying_stats(best_in, best_out, e, near)
    return out


def _copying_stats(best_in, best_out, e, near):
    """Closest-synthetic-person similarity of planted (best_in) against kept-aside patients (best_out)."""
    attack_auc = auc(best_in, best_out)
    hist_in = np.histogram(best_in, e)[0].tolist()
    hist_out = np.histogram(best_out, e)[0].tolist()
    near_in = int((best_in >= near).sum())
    near_out = int((best_out >= near).sum())
    return dict(auc=attack_auc, mean_similarity_in=float(best_in.mean()), mean_similarity_out=float(best_out.mean()),
                similarity_hist=dict(edges=e.tolist(), inserted=hist_in, held_out=hist_out),
                near_copies_in=near_in, near_copies_out=near_out, n_canaries=int(len(best_in)))


def _closest_distance(S, s_sizes, ref, vocab):
    """Jaccard distance of each row of S to its closest person in `ref`, in blocks of 500 synthetic people."""
    Rm = code_matrix(ref, vocab)
    r_sizes = np.asarray(Rm.sum(1)).ravel()
    RT = Rm.T.tocsc()
    best = np.full(S.shape[0], np.inf)
    for s in range(0, S.shape[0], 500):
        inter = (S[s:s + 500] @ RT).toarray()
        union = s_sizes[s:s + 500, None] + r_sizes[None, :] - inter
        # two people with no codes at all are identical, so their similarity is 1
        similarity = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        distance = 1 - similarity
        best[s:s + 500] = distance.min(1)
    return best


def distance_to_closest_record(synth, members, nonmembers, vocab, n_synth=5000, n_ref=20000, seed=0,
                               min_cell=MIN_CELL):
    """DCR check: are synthetic people closer to the training members than to people the model never saw?

    We sample equal numbers of members and non-members, because the distance to the closest record falls with the
    size of the reference set, and compare each synthetic person's Jaccard distance to the closest of each. A
    generator that does not copy puts about half of the people closer to the members; ties are counted
    separately. Shares and histogram bins standing for 1-9 real people are suppressed.

    Parameters
    ----------
    synth, members, nonmembers : data.Cohort
    vocab : data.Vocab
    n_synth, n_ref : int
        Synthetic people and reference people per side.
    seed : int, default 0
    min_cell : int

    Returns
    -------
    dict
        Shares closer and tied, median distances, shares of exact copies, histograms and the sample sizes.
    """
    rng = np.random.default_rng(seed)

    # a random subset kept in the original order, so that the result does not depend on how people were sorted
    def pick(c, n):
        chosen = rng.choice(len(c), min(n, len(c)), replace=False)
        return c.subset(np.sort(chosen))

    S = code_matrix(pick(synth, n_synth), vocab)
    n_ref = min(n_ref, len(members), len(nonmembers))
    s_sizes = np.asarray(S.sum(1)).ravel()
    references = (('members', pick(members, n_ref)), ('nonmembers', pick(nonmembers, n_ref)))
    out = {}
    for name, ref in references:
        out[name] = _closest_distance(S, s_sizes, ref, vocab)
    e = np.linspace(0, 1, 21)

    def counts(name):
        return np.histogram(np.clip(out[name], 0, 1), e)[0]

    def share(m):
        # a share standing for 1-9 people is suppressed; a share of 0 is reported
        if m.sum() >= min_cell or m.sum() == 0:
            return float(m.mean())
        return None

    # exact ties are common, since many people share a short code set, so they are reported apart
    closer = (out['members'] < out['nonmembers']).mean()
    ties = (out['members'] == out['nonmembers']).mean()
    return dict(share_closer_to_members=float(closer), share_ties=float(ties),
                median_dcr_members=float(np.median(out['members'])),
                median_dcr_nonmembers=float(np.median(out['nonmembers'])),
                share_exact_copies_of_members=share(out['members'] == 0),
                share_exact_copies_of_nonmembers=share(out['nonmembers'] == 0),
                histogram=dict(edges=e.tolist(), to_members=suppress_small(counts('members'), min_cell),
                               to_nonmembers=suppress_small(counts('nonmembers'), min_cell)),
                n_synthetic=int(S.shape[0]), n_reference=int(n_ref))


LO, HI = 40.0, 70.0


def window_rate(coh, vocab, lo=LO, hi=HI, cap_events=None):
    """Disease events (codes and death) and person-years between ages lo and hi, summed over the cohort.

    Person-years run from lo to the end of follow-up, at most hi - lo per person. With cap_events each person
    contributes at most that many events, which bounds the sensitivity of the total for the Laplace mechanism.

    Returns
    -------
    events, person_years : float
    """
    lens = coh.lengths()
    rows = np.repeat(np.arange(len(coh)), lens)
    years = coh.age / 365.25
    in_window = (coh.tok >= vocab.first_code) & (years >= lo) & (years < hi)
    per_person = np.bincount(rows[in_window], minlength=len(coh)).astype(float)
    if cap_events:
        per_person = np.minimum(per_person, cap_events)
    end = coh.end_or_last() / 365.25
    person_years = np.clip(end - lo, 0, hi - lo)
    return float(per_person.sum()), float(person_years.sum())


def dp_real_rate(coh, vocab, eps=0.01, cap_events=60, seed=0):
    """The real event rate at 40-70 years, released with the Laplace mechanism (pure epsilon-DP).

    Half of the budget goes to each of the two totals. A person changes the event total by at most cap_events and
    the person-year total by at most HI - LO, which sets the Laplace scales. Both noisy totals are floored at 1
    before the ratio, so that the noise cannot make the rate negative or infinite.

    Parameters
    ----------
    coh : data.Cohort
        The training members.
    vocab : data.Vocab
    eps : float, default 0.01
    cap_events : int, default 60
    seed : int, default 0

    Returns
    -------
    dict
        rate, epsilon, mechanism, cap_events, window_years and the Laplace scales.
    """
    ev, py = window_rate(coh, vocab, cap_events=cap_events)
    # the offset gives the noise a different stream from the other NumPy draws seeded with the same `seed` (the
    # subsets of distance_to_closest_record); the value is arbitrary and is kept so that results reproduce.
    # TODO: draw the noise from a secret seed or a secure generator for a rate meant for release, because anyone
    # who knows the seed can recompute the noise and subtract it (the notebook uses seed 0)
    rng = np.random.default_rng(seed + 991)
    eps_events = eps / 2
    eps_years = eps / 2
    scale_events = cap_events / eps_events
    scale_years = (HI - LO) / eps_years
    # a zero scale (eps or cap_events of 0) would release the exact rate with no noise at all
    assert scale_events > 0 and scale_years > 0, 'Laplace scales must be positive, got %r' % [scale_events, scale_years]
    ev_noisy = ev + rng.laplace(0, scale_events)
    py_noisy = py + rng.laplace(0, scale_years)
    rate = max(ev_noisy, 1.0) / max(py_noisy, 1.0)
    return dict(rate=rate, epsilon=eps, mechanism='Laplace, pure epsilon-DP', cap_events=cap_events,
                window_years=[LO, HI], laplace_scales=[scale_events, scale_years])


def calibrate(model, vocab, real_rate, device, n=50000, iters=6, batch=2048, seed=0, tol=0.01, log=print):
    """Set model.c.log_scale so that the synthetic event rate at 40-70 years matches real_rate.

    The synthetic rate responds more than proportionally to the scale, because more diseases raise the risk of
    further ones, so a plain ratio update overshoots. Every step samples with the same random numbers (a common
    seed), the slope of log(rate) on log(scale) is estimated from the last two steps (secant, clamped to 0.5-4),
    and the scale finally kept is the evaluated one closest to the real rate.

    Parameters
    ----------
    model : AegisModel
        Modified in place (model.c.log_scale).
    vocab : data.Vocab
    real_rate : float
        The noisy real rate (dp_real_rate).
    device : str or torch.device
    n, iters, batch, seed : int
        Synthetic people per step, most steps, sampling batch and seed.
    tol : float, default 0.01
        Stop when |log(real / synthetic)| is below tol.
    log : callable

    Returns
    -------
    list of dict
        log_scale, synthetic_rate and ratio of every step.

    Raises
    ------
    ValueError
        If real_rate is not positive (too few people in the window for this epsilon).
    """
    if not real_rate > 0:
        raise ValueError('calibration: the noisy real rate is not positive (%.4g); too few people in the '
                         'calibration window for this epsilon' % real_rate)
    hist = []
    s = float(model.c.log_scale)
    slope = 1.0
    generator = ModelGenerator(model, device, batch)
    for it in range(iters):
        model.c.log_scale = s
        # the same seed in every round (common random numbers), so two rounds differ only by the scale and the
        # secant slope is not swamped by sampling noise; the offset makes these people differ from a cohort
        # sampled with `seed` itself
        S = generator.sample(vocab, n, seed=seed + 100)
        ev, py = window_rate(S, vocab)
        # no events at all: we treat it as half an event, so that the log ratio stays finite
        sr = max(ev, 0.5) / max(py, 1.0)
        gap = math.log(real_rate / sr)
        hist.append(dict(log_scale=s, synthetic_rate=sr, ratio=real_rate / sr))
        log('  calibration step %d: scale x%.3f, synthetic %.4f vs real %.4f events/person-year (ratio %.3f)'
            % (it + 1, math.exp(s), sr, real_rate, real_rate / sr))
        if abs(gap) < tol:
            break
        # with two distinct scales we can estimate how strongly the rate responds; before that we assume 1
        if len(hist) >= 2 and abs(hist[-1]['log_scale'] - hist[-2]['log_scale']) > SAME_SCALE:
            previous = hist[-2]
            latest = hist[-1]
            d_rate = math.log(latest['synthetic_rate'] / previous['synthetic_rate'])
            d_scale = latest['log_scale'] - previous['log_scale']
            slope = min(4.0, max(0.5, d_rate / d_scale))
        step = gap / slope
        s = s + step
    best = min(hist, key=lambda h: abs(math.log(h['ratio'])))
    if abs(math.log(best['ratio'])) >= tol:
        warnings.warn('calibration stopped after %d steps with the synthetic rate %.1f%% from the real one'
                      % (len(hist), 100 * abs(best['ratio'] - 1)), RuntimeWarning)
    model.c.log_scale = best['log_scale']
    log('  kept scale x%.3f (synthetic/real ratio %.3f; slope %.2f)' % (math.exp(best['log_scale']), best['ratio'],
                                                                       slope))
    return hist
