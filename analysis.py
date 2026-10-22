"""Analysis and figures of the walkthrough notebook, in the manner of Delphi's notebooks and utils.py.

The evaluation follows Delphi: DeLong's AUC within age and sex strata, the model's log-rates at the last record
before a diagnosis (cases) and at one random record per person and 5-year age bracket (controls), observed incidence
per person-year, continuation of a
history with the model's competing exponentials, and an age-adjusted rate ratio. The plotting functions draw the
notebook's figures in Delphi's style; each takes the objects it needs as arguments, so that the notebook shows the
steps of the analysis and this file holds the code that repeats.
"""
import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

from aegis.audit import person_scores
from aegis.data import Packed, iterate, MASK_TIME, NO_EVENT, PAD
from aegis.model import NEG
from aegis.synthetic import copying_test, distance_to_closest_record, U_FLOOR, window_rate
from aegis.train import train
from simulation import first_ages, true_log_rates

YEAR = 365.25
# Floor on the person-years of an age bin, only to avoid dividing by 0: such bins become nan in the same line.
_MIN_PERSON_YEARS = 1e-9
# Two AUCs closer than this count as equal: far below the precision we report, so only float rounding is forgiven.
_SAME_AUC = 1e-6


def delong(case, ctrl):
    """AUC and its variance with DeLong's method (the fast algorithm of Sun and Xu), as in Delphi's evaluation.

    The AUC is the Mann-Whitney probability that a case scores above a control, with ties counted as one half
    through mid-ranks. The variance comes from the structural components of each case and each control, so it
    needs at least two of each.

    Parameters
    ----------
    case : (m,) array_like of float
        Scores of the cases.
    ctrl : (n,) array_like of float
        Scores of the controls.

    Returns
    -------
    auc : float
    variance : float
        DeLong's variance of the AUC.
    """
    m = len(case)
    n = len(ctrl)
    # mid-ranks, so that ties count one half
    rank_case = rankdata(case)
    rank_ctrl = rankdata(ctrl)
    rank_all = rankdata(np.r_[case, ctrl])
    # Structural components: the share of controls below each case, and of cases above each control.
    v_case = (rank_all[:m] - rank_case) / n
    v_ctrl = 1 - (rank_all[m:] - rank_ctrl) / m
    auc = v_case.mean()
    variance = np.var(v_case, ddof=1) / m + np.var(v_ctrl, ddof=1) / n
    return auc, variance


@torch.no_grad()
def scores_for(model, coh, toks, device, max_events=32, batch=512, seed=0, offset=0.1, ages=(40, 80), case_record='last'):
    """The model's log-rates for cases and controls of each disease, as in Delphi's evaluation.

    A case is scored at the last record more than `offset` days before the disease is first recorded. A control
    (a person who never has the disease) is scored at one random record in each 5-year age bracket between the
    ages, as in the paper and Delphi's evaluation; the control records are drawn once per batch and shared by all
    diseases. The batches and the random records depend on `seed` only, never on the model, so two models are scored
    on the same records.

    Parameters
    ----------
    model : AegisModel
    coh : Cohort
        The people to score.
    toks : list of int
        Tokens of the diseases.
    device : str
    max_events : int, default=32
        Records kept per person.
    batch : int, default=512
    seed : int, default=0
        Seed of the no-event tokens and of the random control records.
    offset : float, default=0.1
        Days between the scored record and the diagnosis.
    ages : tuple, default=(40, 80)
        Ages (years) of the scored records.
    case_record : str, default='last'
        'last': the last record before the diagnosis (Delphi and the paper);
        'random_1y': a random record at least 1 year before the diagnosis (a check).

    Returns
    -------
    dict
        For each token: case and control scores (log-rate per day), with the age (years), sex and person index of
        each.
    """
    model.eval()
    packed = Packed(coh, max_events, device)
    rng = np.random.default_rng(seed)
    sex = np.asarray(coh.sex())
    has = _ever_has(coh, toks)

    out = {}
    for k in toks:
        out[k] = dict(case=[], case_age=[], case_sex=[], case_id=[], ctrl=[], ctrl_age=[], ctrl_sex=[], ctrl_id=[])

    tok_tensor = torch.as_tensor(toks, device=device)
    for idx, (x, a, y, b) in iterate(packed, np.arange(packed.n), batch, seed):
        logits = model.logits(x, a, b)[..., tok_tensor].cpu().numpy()  # [people, records, diseases]
        person = idx.cpu().numpy()
        x = x.cpu().numpy()
        a = a.cpu().numpy()
        y = y.cpu().numpy()
        b = b.cpu().numpy()

        # the control records come first and then the cases disease by disease, a fixed order of the draws that
        # gives every model the same records
        ctrl_r, ctrl_p = _control_records(x, a, ages, rng)
        for j, k in enumerate(toks):
            r, p = _case_records(x, a, y, b, k, offset, case_record, rng)
            _add_cases(out[k], logits, j, a, r, p, person, sex, ages)
            never = ~has[k][person[ctrl_r]]
            _add_controls(out[k], logits, j, a, ctrl_r[never], ctrl_p[never], person, sex)

    for k in toks:
        for name in out[k]:
            out[k][name] = np.asarray(out[k][name])
    return out


def _ever_has(coh, toks):
    # at any age, so that a future case is never taken as a control
    has = {}
    for k in toks:
        has[k] = np.zeros(len(coh), bool)
        has[k][coh.rows()[coh.tok == k]] = True
    return has


def _case_records(x, a, y, b, k, offset, case_record, rng):
    r, c = np.nonzero(y == k)
    # Positions are sorted by age with the padding first, so the number of positions before the cut-off, minus
    # one, is the last record before it. The offset keeps out a record made on the day of the diagnosis.
    before = a[r] < (b[r, c] - offset)[:, None]
    p = before.sum(1) - 1
    if case_record == 'random_1y':
        _random_early_records(x, a, b, r, c, p, rng)
    # p = -1 or a padded position: the diagnosis is the first record and there is nothing to score
    ok = (p >= 0) & (x[r, np.maximum(p, 0)] != PAD)
    return r[ok], p[ok]


def _random_early_records(x, a, b, r, c, p, rng):
    # in place on p; a case without a record a year before the diagnosis gets -1 and is dropped by the caller
    for n_case in range(len(r)):
        early = np.flatnonzero((x[r[n_case]] != PAD) & (a[r[n_case]] < b[r[n_case], c[n_case]] - YEAR))
        if len(early) > 0:
            p[n_case] = rng.choice(early)
        else:
            p[n_case] = -1


def _add_cases(scores, logits, j, a, r, p, person, sex, ages):
    age = a[r, p] / YEAR
    keep = (age >= ages[0]) & (age < ages[1])
    scores['case'] += list(logits[r[keep], p[keep], j])
    scores['case_age'] += list(age[keep])
    scores['case_sex'] += list(sex[person[r[keep]]])
    scores['case_id'] += list(person[r[keep]])


def _control_records(x, a, ages, rng, width=5):
    """One random record per person and 5-year age bracket of the batch, the paper's control points.

    Every record between the ages gets a random priority, and in each (person, bracket) we keep the record with the
    lowest one, so a person with many records in a bracket weighs no more than one with a single record there.
    The brackets are those of stratified_auc, computed from the same float32 age in years.

    Returns
    -------
    r, p : ndarray of int
        Row in the batch and position of each control record.
    """
    age = a / YEAR
    r, p = np.nonzero((x != PAD) & (age >= ages[0]) & (age < ages[1]))
    bracket = ((age[r, p] - ages[0]) // width).astype(np.int64)
    priority = rng.random(len(r))
    # sorted by person, then bracket, then priority: the first record of each (person, bracket) is the random one
    order = np.lexsort((priority, bracket, r))
    r, p, bracket = r[order], p[order], bracket[order]
    first = np.r_[True, (r[1:] != r[:-1]) | (bracket[1:] != bracket[:-1])]
    return r[first], p[first]


def _add_controls(scores, logits, j, a, r, p, person, sex):
    # the control records of the people who never have this disease
    scores['ctrl'] += list(logits[r, p, j])
    scores['ctrl_age'] += list(a[r, p] / YEAR)
    scores['ctrl_sex'] += list(sex[person[r]])
    scores['ctrl_id'] += list(person[r])


def stratified_auc(s, width=5, ages=(40, 80), min_cases=2):
    """Age- and sex-stratified AUC: the AUC within each 5-year age bracket and sex, averaged over the brackets and then
    over the sexes, with its DeLong variance.

    Parameters
    ----------
    s : dict
        Scores of one disease from scores_for.
    width : int, default=5
        Width of the age brackets (years).
    ages : tuple, default=(40, 80)
    min_cases : int, default=2
        Fewest cases in a bracket.

    Returns
    -------
    dict
        auc, var (variance) and n_cases; nan and 0 when no bracket has enough cases and controls.
    """
    per_sex = []
    for sx in (0, 1):
        aucs, variances, n = _bracket_aucs(s, sx, width, ages, min_cases)
        # We treat the bracket AUCs as independent, as the paper's code does: the variance of the mean is the sum of
        # the variances over K^2. A control can now appear in several brackets, so this ignores a small positive
        # covariance and the interval is probably a little too narrow.
        if len(aucs) > 0:
            per_sex.append((np.mean(aucs), np.sum(variances) / len(aucs) ** 2, n))

    # A code seen only in one sex (breast or prostate cancer) is averaged over that sex alone.
    if len(per_sex) == 0:
        return dict(auc=np.nan, var=np.nan, n_cases=0)
    auc = np.mean([p[0] for p in per_sex])
    variance = np.sum([p[1] for p in per_sex]) / len(per_sex) ** 2
    n_cases = sum(p[2] for p in per_sex)
    return dict(auc=auc, var=variance, n_cases=n_cases)


def _bracket(s, sx, lo, width):
    in_case = (s['case_sex'] == sx) & (s['case_age'] >= lo) & (s['case_age'] < lo + width)
    in_ctrl = (s['ctrl_sex'] == sx) & (s['ctrl_age'] >= lo) & (s['ctrl_age'] < lo + width)
    return in_case, in_ctrl


def _bracket_aucs(s, sx, width, ages, min_cases):
    # DeLong's variance needs two cases and two controls (ddof=1), so smaller brackets are skipped. Every bracket
    # then weighs the same, one with 2 cases as much as one with 200: should we weight by cases instead?
    aucs, variances, n = [], [], 0
    for lo in range(ages[0], ages[1], width):
        in_case, in_ctrl = _bracket(s, sx, lo, width)
        if in_case.sum() >= min_cases and in_ctrl.sum() >= 2:
            auc, variance = delong(s['case'][in_case], s['ctrl'][in_ctrl])
            aucs.append(auc)
            variances.append(variance)
            n += int(in_case.sum())
    return aucs, variances, n


def adjusted_roc(s, case, ctrl, width=5, ages=(40, 80)):
    """Age- and sex-adjusted ROC curve. Every score becomes its percentile among the controls of the same sex and
    5-year age bracket, and the ROC curve is drawn from these percentiles.

    Parameters
    ----------
    s : dict
        Ages and sexes of the cases and controls (from scores_for).
    case : array
        Scores of the cases.
    ctrl : array
        Scores of the controls.
    width : int, default=5
        Width of the age brackets (years).
    ages : tuple, default=(40, 80)

    Returns
    -------
    fpr, tpr : arrays
        False- and true-positive rates at 201 thresholds.
    """
    pct_case = np.full(len(case), np.nan)
    pct_ctrl = np.full(len(ctrl), np.nan)
    for sx in (0, 1):
        for lo in range(ages[0], ages[1], width):
            in_case, in_ctrl = _bracket(s, sx, lo, width)
            # The same rule as stratified_auc, so that the curve and the AUC use the same brackets.
            if in_case.sum() >= 2 and in_ctrl.sum() >= 2:
                reference = np.sort(ctrl[in_ctrl])
                pct_case[in_case] = _control_percentile(reference, case[in_case])
                pct_ctrl[in_ctrl] = _control_percentile(reference, ctrl[in_ctrl])
    # Records in brackets too small to rank stay nan and are left out of the curve.
    pct_case = pct_case[np.isfinite(pct_case)]
    pct_ctrl = pct_ctrl[np.isfinite(pct_ctrl)]

    thresholds = np.linspace(1, 0, 201)
    fpr = np.array([(pct_ctrl >= u).mean() for u in thresholds])
    tpr = np.array([(pct_case >= u).mean() for u in thresholds])
    return fpr, tpr


def _control_percentile(reference, values):
    # mid-rank percentile among the sorted controls, ties counting half, as in the AUC
    below = np.searchsorted(reference, values, 'left')
    not_above = np.searchsorted(reference, values, 'right')
    return (below + not_above) / 2 / len(reference)


def incidence(coh, k, bins):
    """Observed yearly rate of the first occurrence of token k in age bins: events per person-year at risk.

    Parameters
    ----------
    coh : Cohort
    k : int
        Token of the disease.
    bins : array
        Edges of the age bins (years).

    Returns
    -------
    array
        Rate per year in each bin (nan without person-years).
    """
    rows = coh.rows()
    onset = np.full(len(coh), np.inf)  # years; inf for people without the disease
    has_k = coh.tok == k
    onset[rows[has_k]] = coh.age[has_k] / YEAR
    # person-years at risk: until the disease or the end of follow-up
    stop = np.minimum(onset, coh.end_or_last() / YEAR)
    time_in_bin = np.clip(stop[:, None] - bins[None, :-1], 0, np.diff(bins)[None])
    person_years = time_in_bin.sum(0)
    events = np.histogram(onset[np.isfinite(onset) & (onset <= stop)], bins)[0]
    return np.where(person_years > 0, events / np.maximum(person_years, _MIN_PERSON_YEARS), np.nan)


def pad(histories, device):
    """Left-padded token and age tensors from a list of histories.

    The padding goes on the left so that the last position is every person's latest record, which is where
    continue_histories and learned_ratio read the model's rates.

    Parameters
    ----------
    histories : list
        (tokens, ages in days) of each person.
    device : str

    Returns
    -------
    x, a : tensors [people, records]
    """
    longest = max(len(t) for t, _ in histories)
    x = np.full((len(histories), longest), PAD, np.int64)
    a = np.full((len(histories), longest), MASK_TIME, np.float32)
    for i, (tokens, ages) in enumerate(histories):
        assert len(tokens) == len(ages), 'every record needs a token and an age'
        x[i, longest - len(tokens):] = tokens
        a[i, longest - len(tokens):] = ages
    return torch.as_tensor(x, device=device), torch.as_tensor(a, device=device)


def history(coh, i, before=None):
    """Tokens and ages of person i as plain lists, optionally cut at an age.

    Cutting keeps the records strictly before the age, so a diagnosis made exactly at `before` belongs to the
    future, as for a model that predicts from that age.

    Parameters
    ----------
    coh : Cohort
    i : int
        Index of the person.
    before : float, optional
        Age in years; records at or after it are left out.

    Returns
    -------
    tokens, ages : list
        Tokens and ages in days.
    """
    tokens = coh.tok[coh.off[i]:coh.off[i + 1]]
    ages = coh.age[coh.off[i]:coh.off[i + 1]]
    if before is not None:
        keep = ages < before * YEAR
        tokens = tokens[keep]
        ages = ages[keep]
    return list(tokens), list(ages)


@torch.no_grad()
def continue_histories(model, histories, vocab, device, stop_age=80.0, seed=0, max_steps=60):
    """Continue each history with the model, as in Delphi's sampling.

    Every token k waits an exponential time with rate exp(logit_k) and the first one happens (competing
    exponentials). A disease occurs once; a history stops at death or at stop_age.

    Parameters
    ----------
    model : AegisModel
    histories : list
        (tokens, ages in days) of each person.
    vocab : Vocab
    device : str
    stop_age : float, default=80.0
        Age (years) at which sampling stops.
    seed : int, default=0
        Seed of the torch generator behind every draw.
    max_steps : int, default=60
        Most new records per person, no-event tokens included.

    Returns
    -------
    list
        (tokens, ages) of the new records of each person, without the no-event tokens.
    """
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x, a = pad(histories, device)
    n = x.shape[0]
    seen = _seen_tokens(x, n, vocab, device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    new = [([], []) for _ in range(n)]

    for _ in range(max_steps):
        if not alive.any():
            break
        idx = alive.nonzero(as_tuple=True)[0]
        next_tok, next_age = _next_records(model, x, a, idx, seen, generator, device)

        # A death within stop_age is kept as a record and ends the history; a draw past stop_age only ends it.
        stop = (next_age > stop_age * YEAR) | (next_tok == vocab.death)
        ok = next_age <= stop_age * YEAR
        # A column for everyone keeps the tensors rectangular; people who stopped get padding and are not
        # evaluated again.
        new_tok = torch.full((n,), PAD, dtype=torch.long, device=device)
        new_age = torch.full((n,), MASK_TIME, device=device)
        new_tok[idx[ok]] = next_tok[ok]
        new_age[idx[ok]] = next_age[ok]
        x = torch.cat([x, new_tok[:, None]], 1)
        a = torch.cat([a, new_age[:, None]], 1)
        seen[idx[ok], next_tok[ok]] = True
        # no-event tokens can follow each other, so they are never masked
        seen[:, NO_EVENT] = False
        _keep_records(new, idx, ok, next_tok, next_age)
        alive[idx[stop]] = False
    return new


def _seen_tokens(x, n, vocab, device):
    # First occurrences only: a disease already in the history cannot happen again.
    seen = torch.zeros(n, vocab.size, dtype=torch.bool, device=device)
    seen.scatter_(1, x, True)
    seen[:, [PAD, NO_EVENT]] = False
    return seen


def _next_records(model, x, a, idx, seen, generator, device):
    logits = model.logits(x[idx], a[idx])[:, -1].float()  # log-rate per day at the last record
    logits = logits.masked_fill(seen[idx], NEG)
    # Competing exponentials: token k waits -log(u_k) / exp(logit_k) days and the shortest wait happens. The
    # floor on u keeps -log(u) finite.
    u = torch.rand(logits.shape, generator=generator, device=device).clamp_min(U_FLOOR)
    waiting = -torch.log(u) * torch.exp(-logits)
    dt, next_tok = waiting.min(1)
    # The cap at 80 years only keeps the age finite when the total rate underflows; from a start at 60, a capped
    # wait always lands past stop_age and ends the history.
    next_age = a[idx, -1] + dt.clamp(max=80 * YEAR)
    return next_tok, next_age


def _keep_records(new, idx, ok, next_tok, next_age):
    # the no-event tokens stay in x as inputs but aren't returned
    for i, t, s in zip(idx[ok].tolist(), next_tok[ok].tolist(), next_age[ok].tolist()):
        if t != NO_EVENT:
            new[i][0].append(t)
            new[i][1].append(s)


def rate_ratio(coh, e, o, ages=(40, 80), width=5):
    """Rate ratio of the first occurrence of o after e (against before or without e), adjusted for 5-year attained-age
    bands (Mantel-Haenszel).

    Exposure is time-varying: a person adds person-years to the unexposed group until e and to the exposed group
    after it. Within each band we count the outcomes and person-years of both groups and combine the bands with
    the Mantel-Haenszel estimator.

    Parameters
    ----------
    coh : Cohort
    e : int
        Token of the exposure.
    o : int
        Token of the outcome.
    ages : tuple, default=(40, 80)
        Attained ages (years) covered.
    width : int, default=5
        Width of the age bands (years).

    Returns
    -------
    float
        sum_i(a_i T0_i / T_i) / sum_i(b_i T1_i / T_i) over bands i; nan without an unexposed outcome.
    """
    rows = coh.rows()
    end = coh.end_or_last() / YEAR
    first_e = _first_age(coh, rows, e)
    first_o = _first_age(coh, rows, o)
    # Follow-up stops at the outcome because only first occurrences are counted.
    stop = np.minimum(np.minimum(end, first_o), ages[1])

    numerator, denominator = 0.0, 0.0
    for lo in range(ages[0], ages[1], width):
        events_after, events_before, T1, T0 = _band_counts(first_e, first_o, stop, end, lo, lo + width, ages)
        # an empty band adds nothing (and would divide by 0)
        if T1 + T0 > 0:
            numerator += events_after * T0 / (T1 + T0)
            denominator += events_before * T1 / (T1 + T0)
    # No unexposed outcome in any band: the ratio is undefined, and we return nan rather than inf.
    if denominator > 0:
        return numerator / denominator
    return np.nan


def _first_age(coh, rows, k):
    first = np.full(len(coh), np.inf)
    np.minimum.at(first, rows[coh.tok == k], coh.age[coh.tok == k] / YEAR)
    return first


def _band_counts(first_e, first_o, stop, end, lo, hi, ages):
    # person-years after the exposure (t1) and before or without it (t0) in the band [lo, hi)
    t1 = np.clip(np.minimum(stop, hi) - np.maximum(np.maximum(first_e, lo), ages[0]), 0, None)
    t0 = np.clip(np.minimum(np.minimum(stop, first_e), hi) - lo, 0, None)
    event = (first_o >= lo) & (first_o < hi) & (first_o <= end)
    # An exposure on the same day as the outcome counts as not yet exposed, because we cannot order the two;
    # this can only lower the ratio.
    events_after = (event & (first_e < first_o)).sum()
    events_before = (event & ~(first_e < first_o)).sum()
    return events_after, events_before, t1.sum(), t0.sum()


# Plot style of the Delphi notebooks: fine ticks, dotted grid, bold titles, white-edged markers
STYLE = {
    'figure.dpi': 90, 'font.size': 9,
    'axes.grid': True, 'grid.linestyle': ':', 'grid.linewidth': 0.45,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'xtick.major.size': 1.15, 'ytick.major.size': 1.15, 'xtick.major.width': 0.3, 'ytick.major.width': 0.3,
    'xtick.minor.size': 0.8, 'ytick.minor.size': 0.8, 'xtick.minor.width': 0.3, 'ytick.minor.width': 0.3,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'legend.frameon': False,
    'scatter.edgecolors': 'white', 'lines.markeredgecolor': 'white', 'lines.markeredgewidth': 0.65,
}

# Colours by sex: light for all records, normal for observed rates, dark just before a diagnosis
light_female, normal_female, dark_female = '#FCB6D4', '#F0558B', '#B42C55'
light_male, normal_male, dark_male = '#BAEBE3', '#0FB8A1', '#00574A'
COL = {'non-private': '#4C5B61', 'DP-SGD, eps = 8': '#C8553D', 'observed': '#2A9D8F'}
PRIVATE = 'DP-SGD, eps = 8'


def set_style():
    """Use the plot style of the Delphi notebooks for every figure that follows."""
    plt.rcParams.update(STYLE)


def wrap(text, width=30):
    """Break a long title into lines of at most `width` characters."""
    return '\n'.join(textwrap.wrap(text, width))


def legend_right(ax, **kwargs):
    """Legend outside the plot, on the right."""
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), **kwargs)


def positive(values):
    """Rates of 0 become nan, so that nothing is drawn for them on a log axis."""
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan)


def identity(ax):
    """Dashed diagonal; the x and y limits of the plot must be equal."""
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, c='k', ls=(0, (5, 5)), lw=0.7)


def show_history(tokens, ages, title, names, vocab):
    """Print a history: age in years and record."""
    print(title)
    for t, g in zip(tokens, ages):
        print('  %5.1f  %s' % (g / YEAR, names.get(t, vocab.itos[t])))


def history_table(coh, i, names, vocab, before=None, after=None):
    """A person's records (age in years and name) as a table, optionally only before or from an age in years."""
    tokens, ages = history(coh, i)
    rows = []
    for t, g in zip(tokens, ages):
        if (before is None or g < before * YEAR) and (after is None or g >= after * YEAR):
            rows.append(dict(age=round(g / YEAR, 1), record=names.get(t, vocab.itos[t])))
    return pd.DataFrame(rows)


def plot_training(extras):
    """Validation loss of both models per epoch, and the privacy loss spent by DP-SGD."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for name in extras:
        h = pd.DataFrame(extras[name]['history'])
        ax[0].plot(h.epoch, h.val, '.-', ms=9, c=COL[name], label=name)
    ax[0].set(xlabel='Epoch', ylabel='Validation loss per person', title='Loss curves')
    ax[0].legend()
    h = pd.DataFrame(extras[PRIVATE]['history'])
    ax[1].plot(h.epoch, h.epsilon, '.-', ms=9, c=COL[PRIVATE])
    ax[1].axhline(8, ls='--', c='k', lw=0.8)
    ax[1].set(xlabel='Epoch', ylabel='epsilon spent', title='Privacy loss during DP-SGD')
    plt.tight_layout()
    plt.show()


def privacy_record(extras):
    """The privacy record of the DP-SGD model as a one-row table."""
    p = extras[PRIVATE]['privacy']
    keys = ['epsilon', 'delta', 'noise_multiplier', 'sample_rate', 'expected_batch', 'steps', 'clip']
    return pd.DataFrame([{k: p[k] for k in keys}], index=[PRIVATE])


def pick_participant(test, vocab, min_codes=3):
    """The first test participant with at least min_codes diagnoses before 60 who is followed beyond 75.

    The notebook continues this person's history from 60 and compares it with what happened; we want a few
    diagnoses to condition on and 15 years of observed follow-up to compare with.

    Parameters
    ----------
    test : Cohort
    vocab : Vocab
    min_codes : int, default=3
        Fewest diagnoses before age 60.

    Returns
    -------
    int
        Index of the participant; the last participant, with a RuntimeWarning, when nobody qualifies.
    """
    for i in range(len(test)):
        tokens_before_60 = history(test, i, before=60)[0]
        n_diagnoses = sum(t >= vocab.first_code for t in tokens_before_60)
        if n_diagnoses >= min_codes and test.end[i] > 75 * YEAR:
            break
    else:
        warnings.warn('no test participant has %d diagnoses before 60 and follow-up beyond 75; using the last one'
                      % min_codes, RuntimeWarning)
    return i


def continue_from_60(models, test, i, names, vocab, device, stop_age=80, seed=1):
    """Each model's continuation of person i's history from age 60 (one draw), as a table per model."""
    tok60, age60 = history(test, i, before=60)
    tables = {}
    for name, m in models.items():
        start = [(tok60 + [NO_EVENT], age60 + [60 * YEAR])]
        new_tokens, new_ages = continue_histories(m, start, vocab, device, stop_age=stop_age, seed=seed)[0]
        rows = [dict(age=round(g / YEAR, 1), record=names.get(t, vocab.itos[t])) for t, g in zip(new_tokens, new_ages)]
        tables[name] = pd.DataFrame(rows)
    return pd.concat(tables, names=['continued by'])


def plot_risk_trajectory(models, test, i, codes, names, vocab, device):
    """Each model's 1-year risk of a few diseases at every record of person i; grey lines mark their diagnoses."""
    x, a = pad([history(test, i)], device)
    show_toks = [vocab.stoi[c] for c in codes]
    colours = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
    for axi, name in zip(ax, models):
        with torch.no_grad():
            logits = models[name].logits(x, a)[0].cpu().numpy()
        ages = a[0].cpu().numpy() / YEAR
        real = x[0].cpu().numpy() != PAD
        for t, c in zip(show_toks, colours):
            # the risk of at least one event in a year at a constant rate
            risk = 1 - np.exp(-np.exp(logits[real, t]) * YEAR)
            axi.plot(ages[real], risk, ds='steps-post', marker='.', ms=8, c=c, label=names[t])
        for t, g in zip(*history(test, i)):
            if t >= vocab.first_code:
                axi.axvline(g / YEAR, c='lightgrey', lw=0.6, zorder=0)
        axi.set(yscale='log', xlabel='Age', title=name)
    ax[0].set_ylabel('Predicted 1-year risk')
    legend_right(ax[1])
    plt.tight_layout()
    plt.show()


def plot_waiting_times(models, pk_test, n_people=2000):
    """Expected against observed days to the next record at every record of the first n_people test participants.

    The expected time is 1 / (total rate) of the competing exponentials; the line is the mean observed time in
    bins of the expected time, drawn only for bins with at least 10 records.
    """
    edges = 10 ** np.arange(1, 4.6, 0.25)
    mid_t = np.sqrt(edges[1:] * edges[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for axi, name in zip(ax, models):
        expected_t, observed_t = _waiting_times(models[name], pk_test, n_people)
        average = _binned_mean(expected_t, observed_t, edges)
        # Half a day is added so that waiting times of 0 days appear on the log axis.
        axi.scatter(expected_t, observed_t + 0.5, marker='.', c='lightgrey', edgecolors='none', rasterized=True)
        axi.plot(mid_t, average, '.-', ms=9, c=COL[name], label='average')
        axi.set(xscale='log', yscale='log', xlim=(1, 4e4), ylim=(1, 4e4), aspect='equal',
                xlabel='Expected days to next record', title=name)
        identity(axi)
    ax[0].set_ylabel('Observed days to next record')
    ax[0].legend()
    plt.tight_layout()
    plt.show()


def _waiting_times(model, pk_test, n_people):
    expected_t, observed_t = [], []
    with torch.no_grad():
        for _, (x, a, y, b) in iterate(pk_test, np.arange(min(n_people, pk_test.n)), 500, seed=2):
            logits = model.logits(x, a, b)
            ok = (y != PAD) & (x != PAD)
            # The mean of Exp(sum of rates); logsumexp keeps the sum from overflowing.
            expected = 1 / torch.exp(torch.logsumexp(logits, -1))
            expected_t.append(expected[ok].cpu().numpy())
            observed_t.append((b - a)[ok].cpu().numpy())
    return np.concatenate(expected_t), np.concatenate(observed_t)


def _binned_mean(expected_t, observed_t, edges):
    # A mean over fewer than 10 records is too noisy to draw.
    average = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (expected_t >= lo) & (expected_t < hi)
        if in_bin.sum() >= 10:
            average.append(observed_t[in_bin].mean())
        else:
            average.append(np.nan)
    return average


def record_risks(models, pk_test, test, sel, n_people=4000):
    """Every record of the first n_people test participants with each model's 1-year risk of a few diseases.

    The 1-year risk is 1 - exp(-365.25 * rate per day), the chance of at least one event in a year at the rate
    the model gives at that record. Batches use seed 3, so the no-event records are the same on every run.

    Parameters
    ----------
    models : dict of str to AegisModel
    pk_test : Packed
        The test participants, packed.
    test : Cohort
        The same participants, for their sex.
    sel : list of int
        Tokens of the diseases.
    n_people : int, default=4000

    Returns
    -------
    rec : dict of ndarray
        age (years), sex, person and nxt (the next token) of every record.
    risk : dict of str to ndarray [records, diseases]
        1-year risk under each model.
    """
    sex_test = test.sex()
    rec = dict(age=[], sex=[], person=[], nxt=[])
    risk = {name: [] for name in models}
    with torch.no_grad():
        for idx, (x, a, y, b) in iterate(pk_test, np.arange(min(n_people, pk_test.n)), 500, seed=3):
            r, p = np.nonzero((x != PAD).cpu().numpy())
            who = idx.cpu().numpy()[r]
            rec['age'].append(a.cpu().numpy()[r, p] / YEAR)
            rec['sex'].append(sex_test[who])
            rec['person'].append(who)
            rec['nxt'].append(y.cpu().numpy()[r, p])
            for name, m in models.items():
                logits = m.logits(x, a, b)[..., sel].cpu().numpy()[r, p]
                risk[name].append(1 - np.exp(-np.exp(logits) * YEAR))
    for k in rec:
        rec[k] = np.concatenate(rec[k])
    for name in risk:
        risk[name] = np.concatenate(risk[name])
    return rec, risk


def members_by_sex(members):
    """The training members split by sex (0 female, 1 male), for observed incidence."""
    by_sex = {}
    for sx in (0, 1):
        by_sex[sx] = members.subset(np.flatnonzero(members.sex() == sx))
    return by_sex


def _incidence_panel(axi, rec, risk_j, background, pre, case, t, bins, by_sex):
    """One panel of the age-incidence figure: the clouds of risks, the selected case, and observed incidence."""
    axi.scatter(rec['age'][background], risk_j[background], marker='.', s=50, rasterized=True,
                c=np.array([light_female, light_male])[rec['sex'][background]], label='model, all records')
    axi.scatter(rec['age'][pre], risk_j[pre], marker='.', s=50, rasterized=True,
                c=np.array([dark_female, dark_male])[rec['sex'][pre]], label='model, record before the diagnosis')
    if case is not None:
        trajectory = np.flatnonzero(rec['person'] == rec['person'][case])
        trajectory = trajectory[trajectory <= case]
        axi.plot(rec['age'][trajectory], risk_j[trajectory], ds='steps-post', c='k', marker='.', ms=8, mfc='k',
                 label='one case')
        axi.scatter(rec['age'][case], risk_j[case], marker='.', s=200, c='k', zorder=3)
    axi.stairs(positive(incidence(by_sex[0], t, bins)), bins, color=normal_female, lw=2,
               label='observed incidence, female')
    axi.stairs(positive(incidence(by_sex[1], t, bins)), bins, color=normal_male, lw=2,
               label='observed incidence, male')
    axi.set(yscale='log', ylim=(1e-5, 1), xlim=(0, 85))


def plot_age_incidence(models, rec, risk, test, by_sex, sel, bins, names):
    """Delphi's age-incidence figure: one row per model, one panel per disease.

    Light points are the 1-year risks at a random 5,000 records of people who never have the disease, dark points
    the risks at the record just before a diagnosis; the black line follows, up to the diagnosis, the case with
    the most records, and the steps give the observed incidence of the training members by sex.

    Parameters
    ----------
    models : dict of str to AegisModel
    rec, risk : dict
        Records and 1-year risks from record_risks.
    test : Cohort
    by_sex : dict
        Training members by sex, from members_by_sex.
    sel : list of int
        Tokens of the diseases, in the order of the columns of risk.
    bins : array
        Edges of the age bins (years) of the observed incidence.
    names : dict
        Display name of each token.
    """
    ever = _ever_has(test, sel)
    # The background records are drawn once per disease, so both rows show the same records.
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(len(models), len(sel), figsize=(18, 3.2 * len(models)), sharex=True, sharey=True)
    for j, t in enumerate(sel):
        # the record before a diagnosis is the one whose next token is the disease
        pre = np.flatnonzero(rec['nxt'] == t)
        background = np.flatnonzero(~ever[t][rec['person']])
        # 5,000 points probably show the cloud as well as all of them would, in a lighter figure
        background = rng.choice(background, min(5000, len(background)), replace=False)
        case = _case_to_follow(rec, pre)
        for row, name in enumerate(models):
            _incidence_panel(ax[row, j], rec, risk[name][:, j], background, pre, case, t, bins, by_sex)
            if row == 0:
                ax[row, j].set_title(wrap(names[t]), va='top')
        ax[-1, j].set_xlabel('Age')
    for row, name in enumerate(models):
        ax[row, 0].set_ylabel(name + '\nRate per year')
    legend_right(ax[0, -1])
    plt.tight_layout()
    plt.show()


def _case_to_follow(rec, pre):
    # the case with the most records gives the longest line; argmax takes the first of equals
    case = None
    if len(pre) > 0:
        n_records = np.bincount(rec['person'])
        case = pre[np.argmax(n_records[rec['person'][pre]])]
    return case


def calibration_data(models, pk_test, test, cal_toks, brackets):
    """Rows of (predicted yearly rate, person-years, diagnosed) per model, disease, sex and 5-year age bracket.

    A person enters a bracket when followed into it and free of the disease at its start; the prediction is the
    model's rate at their last record before the bracket, and they are followed until the diagnosis, the end of
    follow-up or the end of the bracket.

    Parameters
    ----------
    models : dict of str to AegisModel
    pk_test : Packed
        The test participants, packed.
    test : Cohort
        The same participants, for diagnoses, sex and end of follow-up.
    cal_toks : list of int
        Tokens of the diseases.
    brackets : list of int
        Lower ages (years) of the 5-year brackets.

    Returns
    -------
    dict
        For each model, a dict from (token, sex, bracket) to an ndarray [people, 3].
    """
    end_test = test.end_or_last() / YEAR
    sex_test = test.sex()
    onset = {}
    for t in cal_toks:
        onset[t] = _first_age(test, test.rows(), t)
    cal = {}
    for name, m in models.items():
        cal[name] = _model_calibration(m, pk_test, onset, end_test, sex_test, cal_toks, brackets)
    return cal


def _model_calibration(m, pk_test, onset, end_test, sex_test, cal_toks, brackets):
    # The no-event tokens count as records, so seed 6 fixes which record is the last before each bracket.
    store = {}
    with torch.no_grad():
        for idx, (x, a, y, b) in iterate(pk_test, np.arange(pk_test.n), 500, seed=6):
            logits = m.logits(x, a, b)[..., cal_toks].cpu().numpy()
            ages = a.cpu().numpy() / YEAR
            real = (x != PAD).cpu().numpy()
            who = idx.cpu().numpy()
            for lo in brackets:
                # Only the history before the bracket, so the prediction never sees what happens within it.
                positions = np.where(real & (ages < lo), np.arange(ages.shape[1]), -1)
                last = positions.max(1)
                _add_bracket(store, lo, last, logits, who, onset, end_test, sex_test, cal_toks)
    cal_m = {}
    for key, parts in store.items():
        cal_m[key] = np.concatenate(parts)
    return cal_m


def _add_bracket(store, lo, last, logits, who, onset, end_test, sex_test, cal_toks):
    for j, t in enumerate(cal_toks):
        # last = -1: no record before the bracket and nothing to predict from, so the person isn't at risk;
        # np.maximum(last, 0) below only keeps their index valid
        at_risk = (last >= 0) & (onset[t][who] >= lo) & (end_test[who] > lo)
        stop = np.minimum(np.minimum(onset[t][who], end_test[who]), lo + 5)
        person_years = stop - lo
        diagnosed = onset[t][who] < np.minimum(end_test[who], lo + 5)
        rate = np.exp(logits[np.arange(len(who)), np.maximum(last, 0), j]) * YEAR  # per year
        for sx in (0, 1):
            keep = at_risk & (sex_test[who] == sx)
            rows = np.c_[rate[keep], person_years[keep], diagnosed[keep]]
            store.setdefault((t, sx, lo), []).append(rows)


def calibration_curve(d, groups=5):
    """Mean predicted and observed yearly rate in `groups` equal groups of predicted rate.

    Parameters
    ----------
    d : ndarray [people, 3]
        Predicted rate, person-years and diagnosed (0 or 1).
    groups : int, default 5

    Returns
    -------
    predicted, observed : ndarray, or None with fewer than 5 diagnoses or 10 people per group
    """
    assert d.shape[1] == 3, 'rows are (rate, person-years, diagnosed)'
    # With fewer diagnoses the observed rates are mostly 0 and the curve says nothing.
    if len(d) < 10 * groups or d[:, 2].sum() < 5:
        return None
    parts = np.array_split(np.argsort(d[:, 0]), groups)
    predicted = np.array([d[g, 0].mean() for g in parts])
    observed = np.array([d[g, 2].sum() / d[g, 1].sum() for g in parts])
    return predicted, positive(observed)


def _bracket_colour(cmap, b, n):
    """Colour of age bracket b of n: darker for older ages."""
    return cmap(0.3 + 0.7 * b / (n - 1))


def plot_calibration(cal, cal_toks, brackets, names):
    """Delphi's calibration figure: model rate against observed rate per age bracket; one figure per model."""
    sexes = [(0, 'Women', normal_female, dark_female), (1, 'Men', normal_male, dark_male)]
    for name in cal:
        fig, ax = plt.subplots(2, len(cal_toks), figsize=(18, 6.6), sharex=True, sharey=True)
        for j, t in enumerate(cal_toks):
            for sx, label, normal, dark in sexes:
                cmap = LinearSegmentedColormap.from_list('age', ['white', normal, dark])
                _calibration_curves(ax[sx, j], cal[name], t, sx, cmap, brackets)
                ax[sx, j].set(xscale='log', yscale='log', xlim=(5e-5, 0.5), ylim=(5e-5, 0.5), aspect='equal')
                identity(ax[sx, j])
                ax[sx, 0].set_ylabel(label + '\nObserved rate [1/yr]')
            ax[0, j].set_title(wrap(names[t]), va='top')
            ax[1, j].set_xlabel('Model rate [1/yr]')
        _calibration_legends(ax, sexes, brackets)
        fig.suptitle(name, fontsize=10, weight='bold', ha='left', x=0.05)
        plt.tight_layout()
        plt.show()


def _calibration_curves(axi, cal_model, t, sx, cmap, brackets):
    for b, lo in enumerate(brackets):
        # A bracket nobody entered has no rows; calibration_curve then returns None and we draw nothing.
        curve = calibration_curve(cal_model.get((t, sx, lo), np.zeros((0, 3))))
        if curve is not None:
            axi.plot(curve[0], curve[1], '.-', ms=8, c=_bracket_colour(cmap, b, len(brackets)))


def _calibration_legends(ax, sexes, brackets):
    for sx, label, normal, dark in sexes:
        cmap = LinearSegmentedColormap.from_list('age', ['white', normal, dark])
        handles = [plt.Line2D([], [], c=_bracket_colour(cmap, b, len(brackets)), marker='.', ms=8,
                              label='%d-%d yrs' % (lo, lo + 5)) for b, lo in enumerate(brackets)]
        ax[sx, -1].legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2)


def auc_table(models, test, toks, names, counts, caused, vocab, device, max_events):
    """Age- and sex-stratified AUC of every disease for both models and for the true rate.

    The cases and controls of the non-private model are also scored with the simulation's true rate, the reference
    AUC on exactly the same people and records (see simulation.true_log_rate for why it is not a strict ceiling).

    Parameters
    ----------
    models : dict of str to AegisModel
        Must contain 'non-private'.
    test : Cohort
    toks : list of int
        Tokens of the diseases (death included).
    names : dict
        Display name of each token.
    counts : array
        Number of training cases of each token.
    caused : set of str
        Codes with a cause beyond age and sex (simulation.caused_codes).
    vocab, device, max_events
        As for scores_for.

    Returns
    -------
    auc : pandas.DataFrame
        One row per disease, sorted by the AUC of the true rate.
    scores : dict
        scores_for of each model.
    truth : dict
        The same cases and controls scored with the true rate.
    """
    scores = {}
    for name, m in models.items():
        scores[name] = scores_for(m, test, toks, device, max_events=max_events)
    rows = []
    truth = {}
    first = first_ages(test)
    for t in toks:
        r = dict(disease=names[t], training_cases=int(counts[t]))
        # Death has causes in the simulation (DEATH_LINKS) although it is not a code.
        r['cause'] = (vocab.itos[t] in caused) or (t == vocab.death)
        _model_aucs(r, scores, models, t)
        # scores_for picks records from its seed alone, so one set of true-rate scores serves both models
        truth[t] = _true_scores(scores['non-private'][t], test, t, vocab, first)
        r['true rate'] = stratified_auc(truth[t])['auc']
        rows.append(r)
    auc = pd.DataFrame(rows)
    auc = auc.sort_values('true rate', ascending=False).reset_index(drop=True)
    return auc, scores, truth


def _model_aucs(r, scores, models, t):
    # The column order of the table follows the order in which we add the keys.
    for name in models:
        s = stratified_auc(scores[name][t])
        r[name] = s['auc']
        # half-width of the 95% interval, from the normal approximation
        r[name + ' ci'] = 1.96 * np.sqrt(s['var'])
        r['test cases'] = s['n_cases']


def _true_scores(s0, test, t, vocab, first):
    truth = dict(s0)
    truth['case'] = true_log_rates(test, s0['case_id'], s0['case_age'], t, vocab, first)
    truth['ctrl'] = true_log_rates(test, s0['ctrl_id'], s0['ctrl_age'], t, vocab, first)
    return truth


def mean_aucs(auc):
    """Mean AUC of each model over all codes and over the codes with a cause, and the mean of the true rate."""
    return pd.DataFrame({
        'all codes': [auc['non-private'].mean(), auc[PRIVATE].mean(), np.nan],
        'codes with a cause': [auc.loc[auc.cause, 'non-private'].mean(), auc.loc[auc.cause, PRIVATE].mean(),
                               auc.loc[auc.cause, 'true rate'].mean()],
    }, index=['non-private', PRIVATE, 'true rate'])


def plot_auc(auc, models):
    """AUC per disease with 95% intervals and the true rate, AUC against training cases, and the cost of
    privacy (the change of the AUC with DP-SGD) against training cases."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 3.8), gridspec_kw={'width_ratios': [3, 1, 1]})
    _auc_per_disease(ax[0], auc, models)
    for name in models:
        ax[1].scatter(auc.training_cases, auc[name], c=COL[name], s=24, linewidth=0.65, label=name)
    ax[1].axhline(0.5, c='k', ls='--', lw=0.75)
    ax[1].set(xscale='log', ylim=(0.2, 1.0), xlabel='Training cases', ylabel='AUC', title='AUC vs number of training cases')
    difference = auc[PRIVATE] - auc['non-private']
    ax[2].scatter(auc.training_cases, difference, c=COL[PRIVATE], s=24, linewidth=0.65)
    ax[2].axhline(0, c='k', ls='--', lw=0.75)
    ax[2].set(xscale='log', xlabel='Training cases', ylabel='AUC change with DP-SGD', title='Cost of privacy')
    ax[1].xaxis.set_minor_formatter(plt.NullFormatter())
    ax[2].xaxis.set_minor_formatter(plt.NullFormatter())
    plt.tight_layout()
    plt.show()


def _auc_per_disease(axi, auc, models):
    xx = np.arange(len(auc))
    axi.scatter(xx, auc['true rate'], marker='_', s=120, c='k', label='true rate',
                zorder=3)
    # The two models sit 0.15 left and right of the disease, so their intervals don't overlap.
    for k, name in enumerate(models):
        axi.errorbar(xx + (k - 0.5) * 0.3, auc[name], yerr=auc[name + ' ci'], fmt='.', ms=10, elinewidth=0.8,
                     c=COL[name], label=name)
    axi.axhline(0.5, c='k', ls='--', lw=0.7)
    axi.set_xticks(xx)
    axi.set_xticklabels(auc.disease, rotation=60, ha='right', fontsize=7)
    for label, has_cause in zip(axi.get_xticklabels(), auc.cause):
        if not has_cause:
            label.set_color('grey')
    axi.set(ylabel='AUC (age and sex stratified)', title='AUC per disease with 95% CI', ylim=(0.2, 1.0))
    axi.legend(loc='upper right', ncol=3)


def plot_roc(codes, truth, scores, auc, names, vocab):
    """Age- and sex-adjusted ROC curves of a few diseases for the true rate and both models."""
    auc_by_name = auc.set_index('disease')
    fig, ax = plt.subplots(1, len(codes), figsize=(17, 3.1), sharey=True)
    for axi, c in zip(ax, codes):
        t = vocab.stoi[c]
        fpr, tpr = adjusted_roc(truth[t], truth[t]['case'], truth[t]['ctrl'])
        axi.plot(fpr, tpr, c='k', lw=1.2, label='true rate')
        for name in scores:
            s = scores[name][t]
            fpr, tpr = adjusted_roc(s, s['case'], s['ctrl'])
            axi.plot(fpr, tpr, c=COL[name], lw=1.5, label=name)
        axi.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='False-positive rate')
        identity(axi)
        axi.set_title(wrap(names[t]), va='top')
        row = auc_by_name.loc[names[t]]
        text = 'AUC %.2f / %.2f\n(true rate %.2f)' % (row['non-private'], row[PRIVATE], row['true rate'])
        axi.text(0.97, 0.03, text, ha='right', va='bottom', fontsize=7, transform=axi.transAxes)
    ax[0].set_ylabel('True-positive rate')
    legend_right(ax[-1])
    plt.tight_layout()
    plt.show()


def learned_ratio(m, test, e, o, device, n=400):
    """Rate ratio of outcome o at age 60 that model m predicts when exposure e is added at 59.5 years.

    For up to n test participants followed beyond 60 without e or o before 60, we compare the model's rate of o at
    60 with and without e added half a year earlier; the ratio is the geometric mean over the people.

    Parameters
    ----------
    m : AegisModel
    test : Cohort
    e, o : int
        Tokens of the exposure and of the outcome.
    device : str
    n : int, default=400
        Most people compared; the first n who qualify, in cohort order.

    Returns
    -------
    float
        exp of the mean difference of the log-rates.
    """
    without_e, with_e = [], []
    for i in range(len(test)):
        tokens, ages = history(test, i, before=60)
        # At least two records, so the model sees more than the sex token.
        if test.end[i] > 60 * YEAR and e not in tokens and o not in tokens and len(tokens) >= 2:
            # we read the rates at a no-event token at 60, the same position in both histories
            without_e.append((tokens + [NO_EVENT], ages + [60 * YEAR]))
            with_e.append((tokens + [e, NO_EVENT], ages + [59.5 * YEAR, 60 * YEAR]))
        if len(without_e) == n:
            break
    with torch.no_grad():
        log_rate_without = m.logits(*pad(without_e, device))[:, -1, o]
        log_rate_with = m.logits(*pad(with_e, device))[:, -1, o]
    return float(torch.exp((log_rate_with - log_rate_without).mean()))


def link_table(models, test, links, diseases, vocab, device):
    """True and learned rate ratio of every simulated link, for each model.

    The learned ratio (learned_ratio) is the change of the model's rate of the outcome at 60 when the exposure
    is added half a year earlier.
    """
    rows = []
    for (e, o), rr in links.items():
        r = dict(link=diseases[e][0] + ' -> ' + diseases[o][0], true=rr)
        for name, m in models.items():
            r[name] = learned_ratio(m, test, vocab.stoi[e], vocab.stoi[o], device)
        rows.append(r)
    return pd.DataFrame(rows)


def kept_link_effect(links):
    """Share of the effect (log rate ratio) of the links the non-private model learned (ratio >= 1.5) that the
    DP-SGD model keeps, and the number of those links."""
    learned = links['non-private'] >= 1.5
    kept = np.nan
    if learned.any():
        kept = np.log(links.loc[learned, PRIVATE]).sum() / np.log(links.loc[learned, 'non-private']).sum()
    return kept, int(learned.sum())


def plot_links(links):
    """Learned against true rate ratio of every link, for both models."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name in (n for n in COL if n in links):
        ax.scatter(links.true, links[name], c=COL[name], s=36, linewidth=0.65, label=name)
    ax.axhline(1, c='grey', lw=0.5)
    ax.set(xscale='log', yscale='log', xlim=(0.5, 30), ylim=(0.5, 30), aspect='equal',
           xlabel='True rate ratio (simulation)', ylabel='Rate ratio learned by the model', title='Learned disease links')
    identity(ax)
    legend_right(ax)
    plt.show()


def no_cause_checks(models, scores, test, toks, caused, names, vocab, device, max_events):
    """Three checks of the AUCs of the codes without a cause.

    For each such code: the AUC as in section 7; with each case scored at a random record at least 1 year before
    the diagnosis instead of the last record; with only the controls followed for at least 5 more years; and the
    rate ratio the non-private model predicts after an unrelated diagnosis (four other codes without a cause).

    Parameters
    ----------
    models, test, caused, names, vocab, device, max_events
        As for auc_table.
    scores : dict
        scores_for of each model, from auc_table.
    toks : list of int
        Tokens of the diseases; death and the codes in `caused` are left out.

    Returns
    -------
    pandas.DataFrame
        One row per code without a cause.
    """
    end_test = test.end_or_last() / YEAR
    no_cause = [t for t in toks if t != vocab.death and vocab.itos[t] not in caused]
    scores_1y = {}
    for name, m in models.items():
        scores_1y[name] = scores_for(m, test, no_cause, device, max_events=max_events, case_record='random_1y')
    rows = []
    for t in no_cause:
        r = dict(disease=names[t])
        r['test cases'] = len(scores['non-private'][t]['case'])
        for name in models:
            s = scores[name][t]
            r[name] = stratified_auc(s)['auc']
            # a record a year before the diagnosis probably holds none of the work-up that led to it
            r[name + ', cases scored 1+ year before'] = stratified_auc(scores_1y[name][t])['auc']
            r[name + ', controls followed 5+ years'] = stratified_auc(_followed_controls(s, end_test))['auc']
        r['rate ratio after an unrelated diagnosis'] = _unrelated_ratio(models['non-private'], test, no_cause, t,
                                                                        device)
        rows.append(r)
    return pd.DataFrame(rows)


def _followed_controls(s, end_test):
    # A control with little follow-up left might be a case we haven't seen yet; keeping only 5+ years of
    # follow-up shows whether such hidden cases move the AUC.
    keep = end_test[s['ctrl_id']] - s['ctrl_age'] >= 5
    s5 = dict(s)
    s5['ctrl'] = s['ctrl'][keep]
    s5['ctrl_age'] = s['ctrl_age'][keep]
    s5['ctrl_sex'] = s['ctrl_sex'][keep]
    s5['ctrl_id'] = s['ctrl_id'][keep]
    return s5


def _unrelated_ratio(model, test, no_cause, t, device):
    # Geometric mean over four unrelated exposures; a model that learned no spurious link gives about 1.
    others = [e for e in no_cause if e != t][:4]
    ratios = [learned_ratio(model, test, e, t, device, n=200) for e in others]
    return np.exp(np.mean(np.log(ratios)))


def reference_models(reference, val, vocab, cfg, device, epsilon=None, n_target=None):
    """Two reference models, each trained with the target's recipe on half of the reference set.

    As in the paper (Methods: Membership attacks), the reference models of a private target are private too, at its
    epsilon and about its sampling rate: we scale the expected batch to the size of the half, so that a person is
    sampled as often per step as in the target, and keep the target's epochs, clipping and delta. Each step then
    averages fewer people, so the reference models are noisier than the target, as the paper's were.

    Parameters
    ----------
    reference, val : Cohort
        The reference set (never in the target's training data) and the validation set.
    vocab : Vocab
    cfg : dict
        The target's training settings (train()).
    device : str
    epsilon : float, optional
        The target's epsilon; None trains non-private reference models.
    n_target : int, optional
        Number of training histories of the target, needed with epsilon to match its sampling rate.

    Returns
    -------
    list of AegisModel
    """
    half = len(reference) // 2
    ref_cfg = dict(cfg)
    if epsilon:
        assert n_target, 'a private target needs n_target to match its sampling rate'
        ref_cfg['dp_batch'] = max(1, round(cfg['dp_batch'] * half / n_target))
    refs = []
    for h in range(2):
        ref_data = reference.subset(np.arange(h * half, (h + 1) * half))
        label = 'reference %d%s' % (h, ', eps = %g' % epsilon if epsilon else '')
        refs.append(train(ref_data, val, vocab, ref_cfg, device, epsilon=epsilon, seed=10 + h, label=label)[0])
    return refs


def audit_packs(members, nonmembers, planted, max_events, device):
    """Members (as many as the non-members), non-members and the planted patients, packed for the audit."""
    msub = members.subset(np.arange(len(nonmembers)))
    packs = {}
    packs['members'] = Packed(msub, max_events, device)
    packs['nonmembers'] = Packed(nonmembers, max_events, device)
    packs['canary_out'] = Packed(planted['out'], max_events, device)
    packs['canary_in1'] = Packed(planted['in1'], max_events, device)
    packs['canary_in8'] = Packed(planted['in8'], max_events, device)
    return packs, msub


def audit_summary(audits, extras):
    """The main numbers of the audit of each model as a table."""
    summary = {}
    for name, r in audits.items():
        claimed = np.nan
        if extras[name]['privacy']:
            claimed = extras[name]['privacy']['epsilon']
        summary[name] = {
            'planted once: AUC': r['canary']['1']['auc'],
            'planted 8x: AUC': r['canary']['8']['auc'],
            'loss attack: AUC': r['loss_attack']['auc'],
            'reference attack: AUC': r['reference_attack']['auc'],
            'psychiatric attack: AUC': r['psychiatric_leakage']['auc'],
            'one-run lower bound on eps': r['one_run_epsilon']['epsilon'],
            'threshold lower bound on eps': r['empirical_epsilon_lower_bound']['best'],
            'claimed eps': claimed,
        }
    return pd.DataFrame(summary)


def plot_audit(audits, models, packs):
    """ROC curves of the attacks on planted patients and of the reference-model attack, and the attack scores of
    the patients planted eight times against the kept-aside ones (artificial patients, shown person by person)."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, r in audits.items():
        roc = r['canary']['8']['roc']
        ax[0].plot(roc['fpr'], roc['tpr'], '-', c=COL[name], label=name + ', planted 8x')
        roc = r['canary']['1']['roc']
        ax[0].plot(roc['fpr'], roc['tpr'], ':', c=COL[name], label=name + ', planted 1x')
        roc = r['reference_attack']['roc']
        ax[1].plot(roc['fpr'], roc['tpr'], c=COL[name], label=name)
    for axi, title in zip(ax[:2], ('Planted patients', 'Reference-model attack (real members)')):
        axi.set(xscale='log', yscale='log', xlim=(5e-3, 1), ylim=(5e-3, 1), aspect='equal',
                xlabel='False-positive rate', ylabel='True-positive rate', title=title)
        identity(axi)
        axi.legend(loc='lower right')
    _planted_histograms(ax[2], audits, models, packs)
    ax[2].set(xlabel='Attack score (minus loss)', ylabel='Density', title='Planted 8x and kept-aside patients')
    legend_right(ax[2])
    plt.tight_layout()
    plt.show()


def _planted_histograms(axi, audits, models, packs):
    for name in models:
        score = audits[name]['canary']['8']['score']
        s8 = -person_scores(models[name], packs['canary_in8'])[score]
        s_out = -person_scores(models[name], packs['canary_out'])[score]
        # nan-aware limits, since a person without targets scores nan
        both = np.r_[s8, s_out]
        edges = np.linspace(np.nanmin(both), np.nanmax(both), 25)
        axi.stairs(np.histogram(s_out, edges, density=True)[0], edges, ls=':', color=COL[name],
                   label=name + ': kept aside')
        axi.stairs(np.histogram(s8, edges, density=True)[0], edges, lw=2, color=COL[name], label=name + ': planted 8x')


def sampling_shares(models, test, toks, vocab, device, n_people=1000):
    """Observed and sampled share of people free of each disease at 60 who get it between 60 and 75.

    We take the first n_people test participants alive at 60 whose outcome by 75 is known: followed to 75, or dead
    between 60 and 75. Keeping the deaths matters, since the sampled histories also stop at death; with survivors
    only, the observed shares would be conditional on surviving to 75 and death would always be 0. People censored
    between 60 and 75 are left out, as we cannot tell what happened to them. Their histories are cut at 60 and
    continued with each model to 75 (one draw each).

    Parameters
    ----------
    models : dict of str to AegisModel
    test : Cohort
    toks : list of int
        Tokens of the diseases.
    vocab : Vocab
    device : str
    n_people : int, default=1000

    Returns
    -------
    share : ndarray
        Observed share for each token.
    sampled : dict of str to ndarray
        Sampled share for each token under each model.
    """
    ids = _outcome_known(test, vocab, 60, 75)[:n_people]
    cut = [history(test, i, before=60) for i in ids]
    observed, at_risk = _observed_counts(test, ids, cut, toks)
    # The floor of 1 only avoids 0/0 for a disease everyone already has.
    share = observed / np.maximum(at_risk, 1)
    sampled = {}
    for name, m in models.items():
        starts = [(tokens + [NO_EVENT], ages + [60 * YEAR]) for tokens, ages in cut]
        continued = continue_histories(m, starts, vocab, device, stop_age=75, seed=4)
        n_new = _new_counts(cut, continued, toks)
        sampled[name] = n_new / np.maximum(at_risk, 1)
    return share, sampled


def _outcome_known(test, vocab, lo, hi):
    """Test participants alive at `lo` years who are followed to `hi` or die between `lo` and `hi`."""
    rows = test.rows()
    death_age = np.full(len(test), np.inf)
    is_death = test.tok == vocab.death
    death_age[rows[is_death]] = test.age[is_death] / YEAR
    end = test.end / YEAR
    keep = (end >= hi) | ((death_age >= lo) & (death_age < hi))
    return [i for i in range(len(test)) if keep[i]]


def _observed_counts(test, ids, cut, toks):
    observed = np.zeros(len(toks))
    at_risk = np.zeros(len(toks))
    for i, (tokens, ages) in zip(ids, cut):
        later = set()
        for t, g in zip(*history(test, i)):
            if 60 * YEAR <= g < 75 * YEAR:
                later.add(t)
        # At risk means free of the disease before 60, the same people the sampled side counts.
        for j, k in enumerate(toks):
            if k not in tokens:
                at_risk[j] += 1
                observed[j] += k in later
    return observed, at_risk


def _new_counts(cut, continued, toks):
    n_new = np.zeros(len(toks))
    for (tokens, ages), (new_tokens, new_ages) in zip(cut, continued):
        for j, k in enumerate(toks):
            if k not in tokens and k in new_tokens:
                n_new[j] += 1
    return n_new


def plot_sampling(share, sampled, toks, vocab):
    """Sampled against observed 15-year incidence; diseases off by more than 3 times are labelled."""
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for name in sampled:
        ax.scatter(share, sampled[name], s=36, linewidth=0.65, c=COL[name], label=name)
        for j in range(len(toks)):
            if share[j] > 0 and sampled[name][j] > 0 and abs(np.log(sampled[name][j] / share[j])) > np.log(3):
                label = 'Death' if toks[j] == vocab.death else vocab.itos[toks[j]]
                ax.annotate(label, (share[j], sampled[name][j]), fontsize=7, color=COL[name], xytext=(4, 2),
                            textcoords='offset points')
    ax.set(xscale='log', yscale='log', xlim=(1e-3, 1), ylim=(1e-3, 1), aspect='equal', xlabel='Observed share, 60-75 years',
           ylabel='Sampled share, 60-75 years', title='Sampled and observed 15-year incidence')
    identity(ax)
    legend_right(ax)
    plt.show()


def plot_synthetic_incidence(synth, by_sex, sel, bins, names):
    """Incidence by sex in the real members (steps) and in each synthetic cohort (points), one row per model."""
    mid = (bins[1:] + bins[:-1]) / 2
    fig, ax = plt.subplots(len(synth), len(sel), figsize=(18, 3.0 * len(synth)), sharex=True, sharey=True)
    for row, name in enumerate(synth):
        S = synth[name]
        synth_female = S.subset(np.flatnonzero(S.sex() == 0))
        synth_male = S.subset(np.flatnonzero(S.sex() == 1))
        for j, t in enumerate(sel):
            axi = ax[row, j]
            axi.stairs(positive(incidence(by_sex[0], t, bins)), bins, color=normal_female, lw=2, label='real, female')
            axi.plot(mid, positive(incidence(synth_female, t, bins)), '.', ms=10, c=dark_female, label='synthetic, female')
            axi.stairs(positive(incidence(by_sex[1], t, bins)), bins, color=normal_male, lw=2, label='real, male')
            axi.plot(mid, positive(incidence(synth_male, t, bins)), '.', ms=10, c=dark_male, label='synthetic, male')
            axi.set(yscale='log', xlim=(0, 85))
            if row == 0:
                axi.set_title(wrap(names[t]), va='top')
        ax[row, 0].set_ylabel(name + '\nRate per year')
    for axi in ax[-1]:
        axi.set_xlabel('Age')
    legend_right(ax[0, -1])
    plt.tight_layout()
    plt.show()


def association_table(members, synth, links, diseases, vocab):
    """Age-adjusted rate ratio of every link in the real members and in each synthetic cohort.

    Each ratio is the Mantel-Haenszel estimate of rate_ratio over 5-year attained-age bands from 40 to 80, so a
    synthetic cohort that keeps the links of the real members gives about the same ratios.
    """
    rows = []
    for (e, o), rr in links.items():
        r = dict(link=diseases[e][0] + ' -> ' + diseases[o][0], true=rr)
        r['real'] = rate_ratio(members, vocab.stoi[e], vocab.stoi[o])
        for name, S in synth.items():
            r['synthetic, ' + name] = rate_ratio(S, vocab.stoi[e], vocab.stoi[o])
        rows.append(r)
    return pd.DataFrame(rows)


def plot_associations(assoc):
    """Rate ratio of each link in the synthetic cohorts against the real members."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name in (n for n in COL if 'synthetic, ' + n in assoc):
        ax.scatter(assoc.real, assoc['synthetic, ' + name], c=COL[name], s=36, linewidth=0.65, label=name)
    ax.set(xscale='log', yscale='log', xlim=(0.5, 50), ylim=(0.5, 50), aspect='equal', xlabel='Rate ratio in real members',
           ylabel='Rate ratio in the synthetic cohort', title='Disease links in synthetic cohorts')
    identity(ax)
    legend_right(ax)
    plt.show()


def diagnoses_per_person(coh, vocab):
    """Mean number of diagnoses and deaths per person; sex and lifestyle tokens are inputs, not diagnoses."""
    is_event = coh.tok >= vocab.first_code
    return np.bincount(coh.rows()[is_event], minlength=len(coh)).mean()


def events_per_person_year(coh, vocab):
    """Diagnoses and deaths per person-year between 40 and 70 years, the window of the rate calibration."""
    events, person_years = window_rate(coh, vocab)
    return events / person_years


def copying_table(synth, planted, members, nonmembers, vocab):
    """Copying test and DCR check of each synthetic cohort, with its diagnoses per person and per person-year.

    Synthetic people are followed to death or 85 years, real members only to 2022 at the latest, so the diagnoses
    per person of a synthetic cohort are higher even when its rates are right; the events per person-year between
    40 and 70 compare the rates themselves.
    """
    cp = {}
    table = {}
    real_diagnoses = diagnoses_per_person(members, vocab)
    real_rate = events_per_person_year(members, vocab)
    for name, S in synth.items():
        cp[name] = copying_test({'1': planted['in1'], '8': planted['in8']}, planted['out'], S, vocab)
        dcr = distance_to_closest_record(S, members, nonmembers, vocab, n_synth=2000, n_ref=2000)
        table[name] = {
            'copying test, planted 8x: AUC': cp[name]['8']['auc'],
            'copying test, planted once: AUC': cp[name]['1']['auc'],
            'DCR: share closer to a member': dcr['share_closer_to_members'],
            'diagnoses per person (real %.2f)' % real_diagnoses: diagnoses_per_person(S, vocab),
            'events per person-year, 40-70 years (real %.3f)' % real_rate: events_per_person_year(S, vocab),
        }
    return pd.DataFrame(table), cp


def aucs_unchanged(m, test, toks, auc, names, device, max_events, column=PRIVATE):
    """Whether the stratified AUCs of model m equal those in `auc`, on the same people, records and controls.

    We use it after recalibration, which adds one constant to every logit and so should leave every ranking, and
    every AUC, as it was. Two nan AUCs (too few cases) count as equal.

    Parameters
    ----------
    m : AegisModel
    test : Cohort
    toks : list of int
    auc : pandas.DataFrame
        The table from auc_table.
    names : dict
    device : str
    max_events : int
    column : str, default=PRIVATE
        Column of `auc` to compare with.

    Returns
    -------
    bool
    """
    auc_by_name = auc.set_index('disease')
    s_cal = scores_for(m, test, toks, device, max_events=max_events)
    for t in toks:
        before = auc_by_name.loc[names[t], column]
        after = stratified_auc(s_cal[t])['auc']
        if not np.isclose(after, before, atol=_SAME_AUC, equal_nan=True):
            return False
    return True


def summary_table(auc, links, audits, extras, cp):
    """The main numbers of the notebook: accuracy, the link effect kept by DP-SGD, and the audit."""
    best = auc.loc[auc.cause, 'true rate']
    names = [n for n in COL if n in audits]
    kept, _ = kept_link_effect(links)
    gain = {}
    # The share of the true rate's gain over chance, summed over codes, so codes the true rate predicts little count
    # less.
    for name in names:
        gain[name] = (auc.loc[auc.cause, name] - 0.5).sum() / (best - 0.5).sum()
    summary = pd.DataFrame({
        'mean AUC, codes with a cause': [auc.loc[auc.cause, n].mean() for n in names],
        'true rate': [best.mean() for n in names],
        "share of the true rate's gain over 0.5": [gain[n] for n in names],
        'mean AUC, all codes': [auc[n].mean() for n in names],
        'share of the learned link effect': [1.0, kept],
        'reference attack AUC': [audits[n]['reference_attack']['auc'] for n in names],
        'planted 8x AUC': [audits[n]['canary']['8']['auc'] for n in names],
        'one-run lower bound on eps': [audits[n]['one_run_epsilon']['epsilon'] for n in names],
        'claimed eps': ['none', '%.2f' % extras[PRIVATE]['privacy']['epsilon']],
        'copying test AUC (8x)': [cp[n]['8']['auc'] for n in names],
    }, index=names)
    return summary.T
