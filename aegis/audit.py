"""Privacy audit (Methods: Membership attacks; Empirical privacy bounds).

1. Planted patients: artificial patients planted once or eight times in the training data against kept-aside
   patients made the same way (they differ only in whether the model saw them).
2. Loss attack on real participants: 10,000 training members against 10,000 non-members (audit split).
3. Reference-model attack: reference models trained like the target on random halves of the reference split
   (people never in the target's training data); score s(x) = (mu_ref(x) - loss(x)) / sd_ref(x) (equation (4)).
4. Psychiatric attack: mean log-probability of the ICD-10 F diagnoses of members against non-members.
5. Lower bounds on epsilon from the patients planted once: the one-run method (Steinke, Nasr & Jagielski 2023) and
   the threshold method with Clopper-Pearson limits (exploratory, as the threshold is chosen on the same data).
6. Membership advantage within groups: share of members caught minus share of non-members flagged, at the
   overall 1% false-positive threshold of the reference-model attack.
Each attack scores a person by the mean total loss or by the mean cross-entropy of their diagnoses and death;
the stronger of the two is reported (P values are doubled for this choice; Methods). Per-person scores stay in
memory; only aggregates are returned, and rates standing for 1-9 real people are not written.
"""
import math
import warnings

import numpy as np
import torch
from scipy.stats import beta as beta_dist
from scipy.stats import binom

from .data import auc, iterate, log, MIN_CELL, NO_EVENT, suppress_small, tpr_at_fpr

SCORES = ('ce', 'loss')
ROC_FPR = np.logspace(-4, 0, 50)
HIST_BINS = 40
# width given to a histogram whose scores are all equal, so that its bins are not empty intervals
FLAT_WIDTH = 1e-6
# floor on the spread of the reference losses: a person whose reference losses agree exactly would otherwise get an
# infinite score
SD_FLOOR = 1e-3
# epsilon searched in [0, EPS_MAX]; an audit that rejects EPS_MAX reports it, since no DP model of interest is there
EPS_MAX = 20.0


@torch.no_grad()
def person_scores(model, pk, batch=1024, seed=5, psych=None):
    """Membership scores of every person under one model.

    For each person we compute the mean total loss over their targets ('loss'), the mean cross-entropy over their
    diagnoses and death with no-event targets left out ('ce', because the no-event tokens are random and carry no
    trace of the person), and, given psychiatric tokens, the mean log-probability of the person's psychiatric
    targets ('psych'). A person without the relevant targets gets nan. The batches use a fixed seed, so every model
    is scored on identical inputs and scores of a target and its reference models can be compared person by person.

    Parameters
    ----------
    model : AegisModel
    pk : data.Packed
    batch : int, default 1024
    seed : int, default 5
    psych : list of int, optional
        Tokens of the psychiatric codes (Vocab.psychiatric()).

    Returns
    -------
    dict of str to ndarray [N]
        'loss', 'ce' and 'psych' per person.
    """
    model.eval()
    # nan by default: a person with no target of a kind has no score of that kind
    loss = np.full(pk.n, np.nan)
    cel = np.full(pk.n, np.nan)
    plp = np.full(pk.n, np.nan)
    is_psych = None
    if psych is not None:
        psych_idx = torch.as_tensor(psych, dtype=torch.long, device=pk.device)
        is_psych = torch.zeros(model.c.vocab_size, dtype=torch.bool, device=pk.device).index_fill_(0, psych_idx, True)
    for idx, (x, a, y, b) in iterate(pk, np.arange(pk.n), batch, seed):
        tl, valid, ce, _ = model.token_losses(x, a, y, b, parts=True)
        i = idx.cpu().numpy()
        mean_loss = (tl.sum(1) / valid.sum(1).clamp(min=1)).cpu().numpy()
        loss[i] = np.where(valid.any(1).cpu().numpy(), mean_loss, np.nan)
        diagnoses = valid & (y != NO_EVENT)
        n_diagnoses = diagnoses.sum(1)
        mean_ce = ((ce * diagnoses).sum(1) / n_diagnoses.clamp(min=1)).cpu().numpy()
        cel[i] = np.where(n_diagnoses.cpu().numpy() > 0, mean_ce, np.nan)
        if is_psych is None:
            continue
        # log-probability of the observed next token
        log_prob = torch.log_softmax(model.logits(x, a, b).float(), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        target_is_psych = is_psych[y]
        n_psych = target_is_psych.sum(1)
        mean_log_prob = ((log_prob * target_is_psych).sum(1) / n_psych.clamp(min=1)).cpu().numpy()
        plp[i] = np.where(n_psych.cpu().numpy() > 0, mean_log_prob, np.nan)
    return dict(loss=loss, ce=cel, psych=plp)


def roc_points(m, n, min_cell=None):
    """ROC curve of an attack as TPRs at fixed log-spaced FPRs from 1e-4 to 1.

    FPRs below 1 / (number of negatives) are left out, because they cannot be measured. For real people (min_cell
    given) a point is written only when at least min_cell members and min_cell non-members lie above its
    threshold, so that no point stands for 1-9 people.

    Parameters
    ----------
    m, n : ndarray of float
        Finite scores of the members (positives) and non-members.
    min_cell : int, optional

    Returns
    -------
    dict
        'fpr' and 'tpr', lists of the same length.
    """
    fprs = ROC_FPR[ROC_FPR >= 1.0 / max(len(n), 1)]
    # the threshold for each FPR is the matching upper quantile of the non-member scores
    thresholds = np.quantile(n, 1 - fprs)
    points = []
    for fpr, t in zip(fprs, thresholds):
        k_m = int((m > t).sum())
        k_n = int((n > t).sum())
        if not min_cell or (k_m >= min_cell and k_n >= min_cell):
            points.append((float(fpr), float(k_m / max(len(m), 1))))
    return dict(fpr=[p for p, _ in points], tpr=[q for _, q in points])


def score_histogram(m, n, min_cell=MIN_CELL, bins=HIST_BINS, lo_hi=(0.5, 99.5)):
    """Histograms of a score for members and non-members on shared bins.

    The bins span the pooled lo_hi percentiles and the tails are folded into the end bins, so that one extreme
    person does not stretch the axis. A bin with fewer than min_cell people is written as None.

    Returns
    -------
    dict
        'edges', 'members', 'nonmembers' (counts or None) and 'min_cell'.
    """
    lo, hi = np.percentile(np.r_[m, n], lo_hi)
    edges = np.linspace(lo, hi if hi > lo else lo + FLAT_WIDTH, bins + 1)

    def counts(v):
        return np.histogram(np.clip(v, edges[0], edges[-1]), edges)[0]

    return dict(edges=edges.tolist(), members=suppress_small(counts(m), min_cell),
                nonmembers=suppress_small(counts(n), min_cell), min_cell=min_cell)


def _boot(m, n, n_boot=200, seed=0):
    """95% bootstrap intervals of the AUC and of the TPR at 1% FPR; members and non-members are resampled
    separately, because they are independent samples."""
    rng = np.random.default_rng(seed)

    def interval(v):
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    samples = [(m[rng.integers(0, len(m), len(m))], n[rng.integers(0, len(n), len(n))]) for _ in range(n_boot)]
    return interval([auc(mb, nb) for mb, nb in samples]), interval([tpr_at_fpr(mb, nb, 0.01) for mb, nb in samples])


def attack_stats(member_score, nonmember_score, min_cell=MIN_CELL, real=True):
    """Summary of one membership attack, a higher score meaning "predicted member".

    We report the AUC and the TPR at 1% and 0.1% FPR, with bootstrap intervals, ROC points and a histogram when
    both sides have at least 10 people. For real people (real=True) a TPR whose numerator would be 1-9 people is
    replaced by its upper bound (min_cell - 1) / n, the UK Biobank small-cell rule; a TPR of 0 is reported.

    Parameters
    ----------
    member_score, nonmember_score : array_like of float
        Scores; non-finite values are dropped.
    min_cell : int
    real : bool
        False for planted patients, who are artificial and need no suppression.

    Returns
    -------
    dict
    """
    # a person without diagnoses has no 'ce' score; the attack cannot score them, so they leave both counts
    m = np.asarray(member_score, float)
    m = m[np.isfinite(m)]
    n = np.asarray(nonmember_score, float)
    n = n[np.isfinite(n)]
    out = dict(auc=auc(m, n), tpr_at_1pct_fpr=tpr_at_fpr(m, n, 0.01), tpr_at_0_1pct_fpr=tpr_at_fpr(m, n, 0.001),
               n_members=int(len(m)), n_nonmembers=int(len(n)))
    for k in ('tpr_at_1pct_fpr', 'tpr_at_0_1pct_fpr'):
        if real and np.isfinite(out[k]) and 0 < round(out[k] * len(m)) < min_cell:
            # 1-9 real people: an upper bound only
            out.update({k: None, k + '_max': float((min_cell - 1) / max(len(m), 1))})
    # below 10 people a side the bootstrap and the ROC points are mostly noise
    if len(m) >= 10 and len(n) >= 10:
        auc_ci, tpr_ci = _boot(m, n)
        out.update(auc_ci=auc_ci, tpr_at_1pct_fpr_ci=tpr_ci)
        out.update(roc=roc_points(m, n, min_cell if real else None), histogram=score_histogram(m, n, min_cell))
    return out


def _best(fn):
    """Run an attack with each score and report the stronger one, keeping both AUCs in 'by_score'."""
    by = {s: fn(s) for s in SCORES}
    # a nan AUC (too few people with the score) never wins; on a tie the first score, 'ce', is kept
    s = max(by, key=lambda k: by[k]['auc'] if np.isfinite(by[k]['auc']) else -1)
    out = dict(by[s], score=s)
    out['by_score'] = {k: {kk: v[kk] for kk in ('auc', 'tpr_at_1pct_fpr', 'auc_ci') if kk in v} for k, v in by.items()}
    return out


def one_run_epsilon(in_score, out_score, delta=0.0, beta=0.05):
    """Lower bound on epsilon from one training run (Steinke, Nasr & Jagielski, NeurIPS 2023).

    Planted and kept-aside patients are drawn i.i.d., so which of them was trained on is a random assignment. The
    auditor guesses 'in' for the k highest scores and 'out' for the k lowest; under epsilon-DP the number of correct
    guesses among the 2k is dominated by Binomial(2k, e^eps / (1 + e^eps)), and the delta term adds at most
    2k * delta to the error. We return the largest epsilon rejected at level beta, Bonferroni-corrected over the
    grid of k, which is a lower bound with confidence 1 - beta.

    Parameters
    ----------
    in_score, out_score : ndarray of float
        Attack scores of the planted and kept-aside patients.
    delta : float, default 0
    beta : float, default 0.05

    Returns
    -------
    dict
        epsilon (nan with fewer than 20 finite scores), the k, correct guesses and guesses behind it, the
        confidence, the number of patients and the k grid.
    """
    s = np.r_[in_score, out_score]
    lab = np.r_[np.ones(len(in_score), bool), np.zeros(len(out_score), bool)]
    ok = np.isfinite(s)
    s, lab = s[ok], lab[ok]
    N = len(s)
    if N < 20:
        return dict(epsilon=float('nan'))
    # mergesort is stable, so ties between scores are broken the same way on every run
    order = np.argsort(-s, kind='mergesort')
    # the k grid: 1% to 20% of the patients, at least 5 guesses a side, at most half of them
    ks = sorted({max(5, int(N * f)) for f in (0.01, 0.025, 0.05, 0.1, 0.2)} & set(range(1, N // 2 + 1)))
    # Bonferroni over the grid, since we report the best k
    level = beta / len(ks)
    best = dict(epsilon=0.0, k=None, correct=None, guesses=None)
    for k in ks:
        # correct guesses: planted among the k highest plus kept-aside among the k lowest
        correct = int(lab[order[:k]].sum() + (~lab[order[-k:]]).sum())
        r = 2 * k
        # the delta term takes its share of the error budget first
        level_k = level - r * delta

        def p_value(eps):
            return binom.sf(correct - 1, r, math.exp(eps) / (1 + math.exp(eps)))

        # with no evidence against epsilon = 0 this k rejects nothing
        if level_k <= 0 or p_value(0.0) > level_k:
            continue
        lo = EPS_MAX if p_value(EPS_MAX) <= level_k else _bisect(p_value, level_k, 0.0, EPS_MAX)
        if lo > best['epsilon']:
            best = dict(epsilon=float(lo), k=int(k), correct=correct, guesses=r)
    best.update(confidence=1 - beta, n_canaries=int(N), k_grid=[int(k) for k in ks])
    return best


def _bisect(p_value, level, lo, hi, steps=60):
    """The largest epsilon in [lo, hi] rejected at `level`, by `steps` halvings of the interval.

    p_value falls as epsilon grows, so we keep the upper half whenever its midpoint is still rejected.
    """
    for _ in range(steps):
        mid = (lo + hi) / 2
        if p_value(mid) <= level:
            lo = mid
        else:
            hi = mid
    return lo


def epsilon_lower_bound(member_score, nonmember_score, delta=0.0, conf=0.95):
    """Threshold lower bound on epsilon (Jagielski et al. 2020), maximised over thresholds.

    For every threshold, in both tails, the lower Clopper-Pearson limit of the TPR and the upper limit of the FPR
    give eps >= log((TPR_low - delta) / FPR_high). Taking the maximum over thresholds chooses the threshold on the
    same data, so this is exploratory and not a 95% bound; the one-run bound is the formal one.

    Parameters
    ----------
    member_score, nonmember_score : ndarray of float
        Scores; non-finite values (people the attack cannot score) are dropped.
    delta : float, default 0
    conf : float, default 0.95
        Two-sided level of each Clopper-Pearson interval.

    Returns
    -------
    float
        The bound (0 when no threshold gives a positive one; nan with fewer than 5 scores on a side).
    """
    m = np.sort(member_score[np.isfinite(member_score)])
    n = np.sort(nonmember_score[np.isfinite(nonmember_score)])
    if len(m) < 5 or len(n) < 5:
        return float('nan')
    alpha, best = 1 - conf, 0.0
    # every observed score is a candidate threshold; between two scores the counts do not change
    for t in np.unique(np.r_[m, n]):
        tp, fp = int((m >= t).sum()), int((n >= t).sum())
        # upper tail (members score high) and lower tail (members score low)
        for tp_, fp_, N1, N0 in ((tp, fp, len(m), len(n)), (len(n) - fp, len(m) - tp, len(n), len(m))):
            # Clopper-Pearson limits from the beta distribution; at 0 or N the limits are the bounds themselves
            tpr_lo = beta_dist.ppf(alpha / 2, tp_, N1 - tp_ + 1) if tp_ > 0 else 0.0
            fpr_hi = beta_dist.ppf(1 - alpha / 2, fp_ + 1, N0 - fp_) if fp_ < N0 else 1.0
            if tpr_lo - delta > 0 and fpr_hi > 0:
                best = max(best, math.log((tpr_lo - delta) / fpr_hi))
    return float(best)


def _wilson(k, n, z):
    """Wilson score interval of a proportion k / n."""
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_diff(k1, n1, k2, n2, z=1.96):
    """Difference of two independent proportions p1 - p2 with Newcombe's hybrid-score 95% interval.

    We use it for the membership advantage (TPR - FPR) because it stays inside [-1, 1] and behaves well when a rate
    is near 0, where the Wald interval fails.

    Returns
    -------
    d, lower, upper : float
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = _wilson(k1, n1, z)
    l2, u2 = _wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


RECORD_GROUPS = [(1, 4, '1-4'), (5, 9, '5-9'), (10, 19, '10-19'), (20, 39, '20-39'), (40, 10 ** 9, '40+')]
RAREST_GROUPS = [(0, 9, '<10'), (10, 99, '10-99'), (100, 999, '100-999'), (1000, 9999, '1k-10k'),
                 (10000, 10 ** 12, '>10k')]


def _advantage_row(kind, lo, hi, label, sm, sn, groups_m, groups_n, thr, min_cell):
    """Membership advantage in one group (people with lo <= groups[kind] <= hi), or None when it is too small."""
    in_m = (groups_m[kind] >= lo) & (groups_m[kind] <= hi) & np.isfinite(sm)
    in_n = (groups_n[kind] >= lo) & (groups_n[kind] <= hi) & np.isfinite(sn)
    caught = int((sm[in_m] > thr).sum())
    flagged = int((sn[in_n] > thr).sum())
    # small-cell rule: neither the group nor the people caught or flagged may be 1-9 (a count of 0 is fine)
    too_small = min(in_m.sum(), in_n.sum()) < min_cell or 0 < caught < min_cell or 0 < flagged < min_cell
    if too_small:
        return dict(label=label, advantage=None, ci=None)
    d, lower, upper = newcombe_diff(caught, int(in_m.sum()), flagged, int(in_n.sum()))
    return dict(label=label, advantage=d, ci=[lower, upper], tpr=caught / in_m.sum(), fpr=flagged / in_n.sum(),
                n_members=int(in_m.sum()), n_nonmembers=int(in_n.sum()))


def membership_advantage(sm, sn, groups_m, groups_n, min_cell=MIN_CELL):
    """Membership advantage (TPR - FPR) of an attack within groups of people, at its overall 1% FPR threshold.

    Groups are defined by the number of diagnoses and by the training count of the person's rarest code, because
    people with long histories or rare codes are the ones a model is most likely to memorise. Each group gets
    Newcombe's interval; a group is reported only with at least min_cell members and non-members and when neither
    rate stands for 1-9 people.

    Parameters
    ----------
    sm, sn : ndarray of float
        Attack scores of the members and non-members.
    groups_m, groups_n : dict
        'records' and 'rarest_code' per person (person_groups).
    min_cell : int

    Returns
    -------
    dict
        threshold_fpr and, for 'records' and 'rarest_code', one row per group.
    """
    thr = float(np.quantile(sn[np.isfinite(sn)], 0.99))
    groups = {}
    for kind, bins in (('records', RECORD_GROUPS), ('rarest_code', RAREST_GROUPS)):
        groups[kind] = [_advantage_row(kind, lo, hi, label, sm, sn, groups_m, groups_n, thr, min_cell)
                        for lo, hi, label in bins]
    return dict(threshold_fpr=0.01, groups=groups)


class Attack:
    """A membership attack: score(group, s) gives each person of a group a membership score, higher meaning
    'in the training data', from the per-person score s ('ce' or 'loss'; see person_scores)."""

    def score(self, group, s):
        raise NotImplementedError

    def against(self, inside, outside, real=True):
        """The attack separating the people of `inside` from those of `outside`, with the stronger score."""
        return _best(lambda s: attack_stats(self.score(inside, s), self.score(outside, s), real=real))


class LossAttack(Attack):
    """Members have a lower loss under the target model (Yeom et al. 2018)."""

    def __init__(self, S):
        self.S = S

    def score(self, group, s):
        return -self.S[group][s]


class ReferenceAttack(Attack):
    """The person's loss under the target model, standardised by their losses under reference models trained
    without them (LiRA-style, Carlini et al. 2022).

    With at least four reference models we use each person's own spread; with fewer, a per-person spread is too
    noisy, so we pool the spread over the non-members.
    """

    def __init__(self, S, R, n_refs):
        self.S, self.R, self.n_refs = S, R, n_refs

    def spread(self, Rk, mu, s):
        if self.n_refs >= 4:
            return np.nanstd(Rk, 0)
        Rn = np.stack([r[s] for r in self.R['nonmembers']])
        return np.full_like(mu, np.nanstd(Rn - np.nanmean(Rn, 0)) + SD_FLOOR)

    def score(self, group, s):
        # people without a diagnosis have no 'ce' score, so the nan means warn on an all-nan column
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            Rk = np.stack([r[s] for r in self.R[group]])
            mu = np.nanmean(Rk, 0)
            return -(self.S[group][s] - mu) / np.maximum(self.spread(Rk, mu, s), SD_FLOOR)


class PsychiatricAttack(Attack):
    """The mean log-probability of the person's psychiatric targets (one score only); it asks whether a model
    reveals who has a mental-health diagnosis, the most sensitive codes of the record."""

    def __init__(self, S):
        self.S = S

    def score(self, group, s='psych'):
        return self.S[group]['psych']

    def against(self, inside, outside, real=True):
        return attack_stats(self.score(inside), self.score(outside), real=real)


def _canary_histograms(out, loss, dups):
    """Coarse score histograms of the planted patients against the kept-aside ones (Supplementary Fig. 7e).

    Patients without a finite score are left out; a set gets a histogram only with at least MIN_CELL patients.
    """
    # one score for all sets, the one chosen for the most repeated set, so the histograms share an axis
    score = out['canary'][str(dups[-1])]['score']
    kept = loss.score('canary_out', score)
    kept = kept[np.isfinite(kept)]
    planted = {}
    for d in dups:
        c = loss.score('canary_in%d' % d, score)
        planted[d] = c[np.isfinite(c)]
    hists = {}
    for d, ci in planted.items():
        # a histogram only for sets of at least MIN_CELL patients
        if min(len(ci), len(kept)) >= MIN_CELL:
            h = score_histogram(ci, kept, bins=10, lo_hi=(1, 99))
            hists[str(d)] = dict(score=score, edges=h['edges'], planted=h['members'], kept_aside=h['nonmembers'])
    return hists


def _epsilon_bounds(out, S, loss, ref, refs, delta):
    """Threshold and one-run lower bounds on epsilon from the patients planted once.

    Both are Bonferroni-corrected over the scores (and attacks) tried, because we report the strongest.
    """
    out['empirical_epsilon_lower_bound'] = dict(
        best=max(epsilon_lower_bound(loss.score('canary_in1', s), loss.score('canary_out', s), delta,
                                     conf=1 - 0.05 / len(SCORES)) for s in SCORES),
        confidence=0.95, canaries_per_arm=int(len(S['canary_out']['ce'])),
        note='maximised over thresholds: exploratory, not a 95% bound')
    attacks = {'': loss, 'ref_': ref} if refs else {'': loss}
    candidates = {p + s: (a.score('canary_in1', s), a.score('canary_out', s)) for p, a in attacks.items() for s in SCORES}
    runs = {k: one_run_epsilon(v[0], v[1], delta, beta=0.05 / len(candidates)) for k, v in candidates.items()}
    strongest = max(runs, key=lambda k: runs[k]['epsilon'] if np.isfinite(runs[k]['epsilon']) else -1)
    out['one_run_epsilon'] = dict(runs[strongest], score=strongest, by_score={k: v['epsilon'] for k, v in runs.items()})


def run_audit(target, refs, packs, vocab, delta, groups=None, batch=1024):
    """The privacy audit of one model: every attack of the paper and the lower bounds on epsilon.

    Parameters
    ----------
    target : AegisModel
        The model under audit.
    refs : list of AegisModel
        Reference models trained without the members (may be empty; then the reference attack is skipped).
    packs : dict of str to data.Packed
        'members', 'nonmembers', 'canary_out', and 'canary_in1' / 'canary_in8' for the patients planted once and
        eight times.
    vocab : data.Vocab
    delta : float
        The delta of the model's guarantee, used in the bounds on epsilon.
    groups : dict, optional
        person_groups of 'members' and 'nonmembers', for the membership advantage.
    batch : int, default 1024

    Returns
    -------
    dict
        canary, loss_attack, reference_attack, canary_reference, membership_advantage, psychiatric_leakage,
        canary_histograms, empirical_epsilon_lower_bound and one_run_epsilon (each when it can be computed).
    """
    psych = vocab.psychiatric()
    # psychiatric scores only for the two groups the psychiatric attack compares
    S = {k: person_scores(target, p, batch, psych=psych if k in ('members', 'nonmembers') else None)
         for k, p in packs.items()}
    # how often each planted set was repeated in training: 'canary_in8' -> 8
    dups = sorted(int(k[len('canary_in'):]) for k in packs if k.startswith('canary_in'))
    loss = LossAttack(S)
    out = dict(canary={str(d): loss.against('canary_in%d' % d, 'canary_out', real=False) for d in dups},
               loss_attack=loss.against('members', 'nonmembers'))
    ref = None
    # the reference attack needs reference models; without them only the loss attack runs
    if refs:
        ref = ReferenceAttack(S, {k: [person_scores(r, p, batch) for r in refs] for k, p in packs.items()}, len(refs))
        out['reference_attack'] = dict(ref.against('members', 'nonmembers'), n_reference_models=len(refs))
        out['canary_reference'] = {str(d): ref.against('canary_in%d' % d, 'canary_out', real=False) for d in dups}
    # the membership advantage uses the reference attack with the score it chose for the members
    if refs and groups is not None:
        s = out['reference_attack']['score']
        out['membership_advantage'] = membership_advantage(ref.score('members', s), ref.score('nonmembers', s),
                                                           groups['members'], groups['nonmembers'])
    out['psychiatric_leakage'] = PsychiatricAttack(S).against('members', 'nonmembers')
    if dups:
        out['canary_histograms'] = _canary_histograms(out, loss, dups)
    # the bounds on epsilon assume each patient was in training once, so they use the patients planted once
    if 1 in dups:
        _epsilon_bounds(out, S, loss, ref, refs, delta)
    log('  audit: planted %s; loss attack AUC %.3f; reference attack AUC %s; psychiatric AUC %.3f; one-run eps >= %s'
        % (', '.join('x%s %.2f' % (k, v['auc']) for k, v in out['canary'].items()), out['loss_attack']['auc'],
           '%.3f' % out['reference_attack']['auc'] if 'reference_attack' in out else 'n/a',
           out['psychiatric_leakage']['auc'],
           '%.2f' % out['one_run_epsilon']['epsilon'] if 'one_run_epsilon' in out else 'n/a'))
    return out


def person_groups(coh, vocab, train_counts):
    """Per person: the number of diagnoses (codes, not death) and the training count of their rarest code."""
    rows = coh.rows()
    is_diagnosis = (coh.tok >= vocab.first_code) & (coh.tok != vocab.death)
    rarest = np.full(len(coh), np.inf)
    np.minimum.at(rarest, rows[is_diagnosis], train_counts[coh.tok[is_diagnosis]].astype(float))
    return dict(records=np.bincount(rows[is_diagnosis], minlength=len(coh)), rarest_code=rarest)
