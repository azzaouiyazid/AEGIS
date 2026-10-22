"""Disease histories, the fixed vocabulary, data splits, planted patients and batching.

Methods: Disease histories; Data splits and planted patients.

The vocabulary is fixed and public (Methods: Disease histories). Delphi builds its vocabulary from the training
data, so its list of codes reveals which rare codes occur in the cohort. AEGIS fixes the vocabulary before any data
are read: all 1,700 three-character ICD-10 codes from A00 to Q99 (chapters I-XVII, the code space Delphi-2M
models), death and 13 special tokens, 1,714 tokens in all.

Token layout (the special tokens are those of Delphi):
  0 pad | 1 no event | 2 female | 3 male | 4-6 BMI low/mid/high | 7-9 smoking never or former/occasional/current
  10-12 alcohol low/mid/high | 13 ... 1712 ICD-10 codes A00 ... Q99 | 1713 death

A cohort holds one disease history per participant (tokens and ages in days), built from UK Biobank
first-occurrence dates, the cancer registry and death records. The splits are made by person, and the planted
patients are artificial histories placed in the training data once or eight times.

Batching follows Delphi's utils.get_batch, vectorised in torch. For a batch of people we add no-event tokens at
random ages (20 uniform draws over 0-100 years, one per five years on average) up to the person's last record, so
that the model also learns periods without disease; during training the ages of the lifestyle tokens are shifted at
random by -20 to +40 years, as in Delphi; the sequence is left-padded and split into inputs (x, ages a) and
next-token targets (y, ages b). Every batch is a deterministic function of (people, seed), so models compared with
each other see identical inputs.
"""
import time
import warnings

import numpy as np
import torch
from scipy.stats import rankdata

# UK Biobank output rule: no count, rate or share may stand for 1-9 people
MIN_CELL = 10


def suppress_small(counts, min_cell=MIN_CELL):
    """The counts as ints, with every count below min_cell (0 included) replaced by None: the MIN_CELL rule."""
    return [int(x) if x >= min_cell else None for x in counts]


def log(msg):
    """Print a message with the time of day, flushed at once so that long runs can be followed in a log file."""
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def device_of(name=None):
    """The torch device called `name`, or the first GPU when there is one and the CPU otherwise."""
    if name:
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def auc(pos, neg):
    """Mann-Whitney AUC: the probability that a positive scores higher than a negative, ties counting one half.

    We compute it from the rank sum of the positives, which is exact with ties and needs one sort. Scores that are
    not finite are dropped first (an attack score can be nan for a person without targets); the result is nan when
    fewer than two positives or two negatives remain, because an AUC from one person on a side says nothing.

    Parameters
    ----------
    pos, neg : array_like of float
        Scores of the positives (members, planted patients, cases) and of the negatives.

    Returns
    -------
    float
        The AUC in [0, 1], or nan.
    """
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float('nan')
    n_pos = len(pos)
    n_neg = len(neg)
    ranks = rankdata(np.r_[pos, neg])
    rank_sum = ranks[:n_pos].sum()
    U = rank_sum - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


def tpr_at_fpr(pos, neg, fpr):
    """True-positive rate at the threshold that flags a share `fpr` of the negatives.

    The threshold is the (1 - fpr) quantile of the negative scores and a positive counts when it is strictly above
    it, the usual reading of "TPR at 1% FPR" in membership inference (Carlini et al. 2022).

    Parameters
    ----------
    pos, neg : array_like of float
        Scores of the positives and of the negatives.
    fpr : float
        False-positive rate in (0, 1).

    Returns
    -------
    float
        The true-positive rate, or nan when either side is empty.
    """
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(neg) == 0 or len(pos) == 0:
        return float('nan')
    threshold = np.quantile(neg, 1 - fpr)
    flagged = pos > threshold
    return float(flagged.mean())


SPECIAL = ['<pad>', '<no_event>', '<female>', '<male>',
           'BMI_low', 'BMI_mid', 'BMI_high',
           'Smoking_low', 'Smoking_mid', 'Smoking_high',
           'Alcohol_low', 'Alcohol_mid', 'Alcohol_high']
PAD, NO_EVENT, FEMALE, MALE = 0, 1, 2, 3
LIFESTYLE = list(range(4, 13))
# inputs the model is never asked to predict, as in Delphi: sex and lifestyle are measured, not events
INPUT_ONLY = [PAD, FEMALE, MALE] + LIFESTYLE
# first letters of the ICD-10 chapters I-XVII
LETTERS = 'ABCDEFGHIJKLMNOPQ'

CHAPTERS = [
    ('I Infectious', 'A00', 'B99'), ('II Neoplasms', 'C00', 'D48'),
    ('III Blood & immune', 'D50', 'D89'), ('IV Endocrine & metabolic', 'E00', 'E90'),
    ('V Mental & behavioural', 'F00', 'F99'), ('VI Nervous system', 'G00', 'G99'),
    ('VII Eye', 'H00', 'H59'), ('VIII Ear', 'H60', 'H95'), ('IX Circulatory', 'I00', 'I99'),
    ('X Respiratory', 'J00', 'J99'), ('XI Digestive', 'K00', 'K93'), ('XII Skin', 'L00', 'L99'),
    ('XIII Musculoskeletal', 'M00', 'M99'), ('XIV Genitourinary', 'N00', 'N99'),
    ('XV Pregnancy', 'O00', 'O99'), ('XVI Perinatal', 'P00', 'P96'),
    ('XVII Congenital', 'Q00', 'Q99')]

# tokens outside the ICD-10 chapters
OWN_CHAPTER = {'Death': 'Death'}


def chapter_of(code):
    """ICD-10 chapter of a three-character code ('I Infectious' ... 'XVII Congenital'), 'Death', or 'Other'.

    Codes such as D49 or H96 fall between the chapter ranges listed in CHAPTERS and are 'Other'.
    """
    return OWN_CHAPTER.get(code) or next((name for name, lo, hi in CHAPTERS if lo <= code <= hi), 'Other')


class Vocab:
    """The fixed public vocabulary: 13 special tokens, the 1,700 codes A00-Q99 and death (1,714 tokens).

    It is built from the ICD-10 code space alone, never from data, so it reveals nothing about which codes occur
    in the cohort. `stoi` and `itos` map between token strings and indices.
    """

    def __init__(self):
        self.codes = ['%s%02d' % (letter, n) for letter in LETTERS for n in range(100)]
        self.itos = SPECIAL + self.codes + ['Death']
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.size = len(self.itos)
        self.first_code = len(SPECIAL)
        self.death = self.size - 1

    def code_tokens(self):
        return list(range(self.first_code, self.death))

    def disease_tokens(self):
        """Tokens evaluated for AUC: the ICD-10 codes and death."""
        return self.code_tokens() + [self.death]

    def chapter(self, tok):
        return chapter_of(self.itos[tok]) if tok >= self.first_code else 'Technical'

    def psychiatric(self):
        return [self.stoi[c] for c in self.codes if c[0] == 'F']

    def hierarchy(self):
        """Block and chapter id of every token, for the hierarchical embeddings (Methods: Model).

        A block is the code's letter and tens digit (for example I20-I29), a close proxy for the ICD-10 blocks that
        needs no lookup table.
        Special tokens and death each get their own block and chapter, so they share no vector with any code.
        TODO: use the official ICD-10 block table; C00-C14 spans two tens digits, and I20-I25 shares I2x with
        I26-I28.

        Returns
        -------
        blocks, chapters : list of int
            Block id and chapter id of each of the 1,714 tokens.
        """
        blocks, chapters, block_ids, chapter_ids = [], [], {}, {}
        for i, t in enumerate(self.itos):
            if self.first_code <= i < self.death:
                block, chapter = t[:2], chapter_of(t)
            else:
                block = chapter = '<%s>' % t
            blocks.append(block_ids.setdefault(block, len(block_ids)))
            chapters.append(chapter_ids.setdefault(chapter, len(chapter_ids)))
        return blocks, chapters


SPLITS = dict(members=0.50, val=0.05, test=0.15, audit=0.10, reference=0.20)


class Cohort:
    """Disease histories in compressed form, one per person.

    Person i owns tok[off[i]:off[i+1]] and age[off[i]:off[i+1]], ages in days and ascending; end[i] is the age in
    days at the end of follow-up (death or censoring) and may be missing. Each disease appears once per person
    because UK Biobank first-occurrence fields give the first date only, so token counts are counts of people.

    Parameters
    ----------
    tok, age : array_like
        Tokens (int32) and ages in days (float32) of all records, person after person.
    off : array_like of int
        Offsets of the people, length n + 1, starting at 0.
    eid, end : array_like, optional
        Participant ids and ages at the end of follow-up.
    """

    def __init__(self, tok, age, off, eid=None, end=None):
        self.tok = np.asarray(tok, np.int32)
        self.age = np.asarray(age, np.float32)
        self.off = np.asarray(off, np.int64)
        self.eid = None if eid is None else np.asarray(eid, np.int64)
        self.end = None if end is None else np.asarray(end, np.float32)
        assert self.tok.shape == self.age.shape, 'every record needs a token and an age'

    def __len__(self):
        return len(self.off) - 1

    def lengths(self):
        return np.diff(self.off)

    def rows(self):
        """Person index of every token."""
        return np.repeat(np.arange(len(self)), self.lengths())

    def subset(self, idx):
        """The people at the indices `idx`, in that order, as a new cohort."""
        idx = np.asarray(idx, np.int64)
        lens = self.lengths()[idx]
        starts = self.off[idx]
        new_off = np.r_[0, np.cumsum(lens)]
        # we gather every record of the chosen people in one fancy index instead of a loop over people
        shift = starts - new_off[:-1]
        within = np.arange(lens.sum())
        pos = np.repeat(shift, lens) + within
        eid = None if self.eid is None else self.eid[idx]
        end = None if self.end is None else self.end[idx]
        return Cohort(self.tok[pos], self.age[pos], new_off, eid, end)

    def repeat(self, k):
        return Cohort.concat([self] * k)

    def drop_tokens(self, toks):
        """The same people without some input tokens (for example the lifestyle inputs)."""
        keep = ~np.isin(self.tok, toks)
        n = np.bincount(self.rows()[keep], minlength=len(self))
        return Cohort(self.tok[keep], self.age[keep], np.r_[0, np.cumsum(n)], self.eid, self.end)

    @staticmethod
    def concat(parts):
        """One cohort from several, people in order; empty parts are skipped.

        Ends of follow-up are kept when any part has them, using the last record for the parts that do not
        (planted patients have no end), so that a mixed training set still has an end for everyone.
        """
        parts = [p for p in parts if len(p)]
        lens = np.concatenate([p.lengths() for p in parts])
        # ids only make sense when every part has them; planted patients have none
        eid = np.concatenate([p.eid for p in parts]) if all(p.eid is not None for p in parts) else None
        end = np.concatenate([p.end_or_last() for p in parts]) if any(p.end is not None for p in parts) else None
        return Cohort(np.concatenate([p.tok for p in parts]), np.concatenate([p.age for p in parts]),
                      np.r_[0, np.cumsum(lens)], eid, end)

    @staticmethod
    def from_long(person, tok, age):
        """A cohort from records in long format (one row per record, any order).

        Records are sorted by person, then age, then token, so that records on the same day always come in the same
        order and every batch built from the cohort is reproducible.

        Parameters
        ----------
        person : array_like of int
            Person index 0..N-1 of each record.
        tok, age : array_like
            Token and age in days of each record.
        """
        order = np.lexsort((tok, age, person))
        person = person[order]
        tok = tok[order]
        age = age[order]
        n_people = int(person.max()) + 1 if len(person) else 0
        counts = np.bincount(person, minlength=n_people)
        off = np.r_[0, np.cumsum(counts)]
        return Cohort(tok, age, off)

    def end_or_last(self):
        """End of follow-up; the age of the last record where it is unknown or earlier than that record."""
        last = np.zeros(len(self), np.float32)
        has_records = self.lengths() > 0
        last[has_records] = self.age[self.off[1:][has_records] - 1]
        return last if self.end is None else np.maximum(self.end, last)

    def token_counts(self, vocab_size):
        """Number of people with each token (first occurrences only, so tokens are people)."""
        return np.bincount(self.tok, minlength=vocab_size)

    def sex(self):
        """0 = female, 1 = male, per person; a person without a sex token counts as male."""
        female = np.zeros(len(self), bool)
        female[self.rows()[self.tok == FEMALE]] = True
        return np.where(female, 0, 1)

    def save(self, path):
        np.savez(path, tok=self.tok, age=self.age, off=self.off,
                 eid=self.eid if self.eid is not None else np.zeros(0, np.int64),
                 end=self.end if self.end is not None else np.zeros(0, np.float32))

    @staticmethod
    def load(path):
        z = np.load(path)
        eid = z['eid'] if len(z['eid']) else None
        end = z['end'] if 'end' in z.files and len(z['end']) else None
        return Cohort(z['tok'], z['age'], z['off'], eid, end)


def make_splits(n, seed=0, fractions=SPLITS):
    """Random split of n people into the five sets of the paper.

    members train the model, val stops training early, test measures accuracy, audit holds the non-members of the
    attacks and reference trains the reference models. We split by person, never by record, because a person's
    records are not independent; the last set takes the remainder so that no one is lost to rounding.

    Parameters
    ----------
    n : int
        Number of people.
    seed : int, default 0
    fractions : dict, default SPLITS
        Share of the people in each set, in order.

    Returns
    -------
    dict of str to ndarray
        Sorted person indices of each set.
    """
    perm = np.random.default_rng(seed).permutation(n)
    out, start, names = {}, 0, list(fractions)
    for i, k in enumerate(names):
        end = n if i == len(names) - 1 else start + int(round(fractions[k] * n))
        out[k] = np.sort(perm[start:end])
        start = end
    return out


def planted_pool(train_counts, kind='rare', length=8):
    """Codes that the planted patients are drawn from.

    'rare' (the paper) takes the codes present in the training members with at most the 30th percentile of the
    positive counts (371 codes in UK Biobank), because rare codes make a planted patient stand out, the hardest case
    for privacy. This pool depends on the training data, so the DP guarantee of a model trained with these patients
    holds for a fixed planted set (Methods). 'any' takes every code of the public vocabulary, chosen without the data.

    Parameters
    ----------
    train_counts : ndarray of int
        Number of training members with each token (Cohort.token_counts).
    kind : {'rare', 'any'}
    length : int
        Codes per planted patient; the pool keeps at least three times as many codes.

    Returns
    -------
    ndarray of int
        Tokens of the pool.
    """
    return PLANTED_POOLS[kind](np.array(Vocab().code_tokens()), train_counts, length)


def _any_codes(codes, train_counts, length):
    """Every code of the public vocabulary, chosen without the data."""
    return codes


def _rare_codes(codes, train_counts, length):
    """The codes present in the training members with at most the 30th percentile of the positive counts."""
    counts = train_counts[codes]
    in_members = counts > 0
    present = codes[in_members]
    if in_members.any():
        max_count = max(1, int(np.percentile(counts[in_members], 30)))
    else:
        max_count = 1
    pool = codes[in_members & (counts <= max_count)]
    # with few members (the demo) the percentile rule can leave too few codes to draw distinct ones from, so we
    # fall back to the rarest present codes
    if len(pool) < length * 3:
        rarest_first = np.argsort(train_counts[present])
        pool = present[rarest_first[:max(length * 3, 1)]]
    return pool


# kind of planted pool -> the function that chooses its codes
PLANTED_POOLS = {'rare': _rare_codes, 'any': _any_codes}


def make_planted(pool, n, length, rng):
    """Artificial patients for the privacy audit.

    Each has a random sex at birth and `length` different codes from `pool` at sorted random ages of 30-75 years.
    They are not real people, so their scores can be shown person by person, and patients planted in the training
    data differ from kept-aside ones only in whether the model saw them.

    Parameters
    ----------
    pool : ndarray of int
        Codes to draw from (planted_pool).
    n, length : int
        Number of patients and codes per patient.
    rng : numpy.random.Generator

    Returns
    -------
    Cohort
    """
    P, K, A = [], [], []
    for i in range(n):
        toks = rng.choice(pool, length, replace=False)
        ages_years = rng.uniform(30, 75, length)
        ages = np.sort(ages_years) * 365.25
        sex = rng.choice([FEMALE, MALE])
        P += [i] * (length + 1)
        K += [sex] + list(toks)
        A += [0.0] + list(ages)
    return Cohort.from_long(np.array(P), np.array(K, np.int32), np.array(A, np.float32))


# age given to padding: far below any real age, so that padding sorts first and never counts as a record
MASK_TIME = -10000.0
NO_EVENT_RATE_YEARS = 5.0


class Packed:
    """A cohort as padded tensors on a device, ready for batching.

    T [N, E] holds the tokens (0 = pad), A [N, E] the ages in days and END [N] the end of follow-up. Each person
    keeps the first `max_events` tokens (128 in the paper). For a person cut at max_events the end of follow-up
    becomes the age of the last kept record, because the model must not be told that nothing happened in time it
    did not see.

    Parameters
    ----------
    coh : Cohort
    max_events : int
    device : str or torch.device
    """

    def __init__(self, coh, max_events, device):
        lens = coh.lengths()
        E = int(min(max_events, max(1, lens.max() if len(lens) else 1)))
        n = len(coh)
        keep = np.minimum(lens, E)
        rows = np.repeat(np.arange(n), keep)
        within = np.arange(keep.sum()) - np.repeat(np.r_[0, np.cumsum(keep)[:-1]], keep)
        src = np.repeat(coh.off[:-1], keep) + within
        T = np.zeros((n, E), np.int64)  # (people, records)
        A = np.full((n, E), MASK_TIME, np.float32)  # days
        T[rows, within] = coh.tok[src]
        A[rows, within] = coh.age[src]
        self.T = torch.from_numpy(T).to(device)
        self.A = torch.from_numpy(A).to(device)
        self.n, self.E, self.device = n, E, device
        self.truncated = int((lens > E).sum())
        if self.truncated:
            warnings.warn('%d of %d histories are longer than max_events=%d and were cut after their first %d records'
                          % (self.truncated, n, E, E), RuntimeWarning)
        self.END = torch.from_numpy(self._end_of_follow_up(coh, A, lens, keep, E)).to(device)

    @staticmethod
    def _end_of_follow_up(coh, A, lens, keep, E):
        """End of follow-up per person, cut back to the last kept record for people longer than E."""
        n = len(coh)
        end = coh.end_or_last().astype(np.float32)
        last_index = np.maximum(keep - 1, 0)
        last_kept_age = A[np.arange(n), last_index]
        last_kept = np.where(keep > 0, last_kept_age, 0.0)
        cut = lens > E
        end = np.where(cut, last_kept, np.maximum(end, last_kept))
        return end.astype(np.float32)


def make_batch(pk, idx, gen, train=True, censor=False):
    """Inputs and next-token targets for the people `idx`, as in Delphi's get_batch.

    We add no-event tokens at random ages up to the last record (or to the end of follow-up with censor=True, which
    we use to measure calibration over the whole follow-up), shift the lifestyle ages during training, sort every
    sequence by age and cut the common left padding. The random draws come from `gen` only, so a batch is a
    function of (people, seed).

    Parameters
    ----------
    pk : Packed
    idx : torch.Tensor of int
        Person indices of the batch.
    gen : torch.Generator
    train : bool, default True
        Shift the lifestyle ages (training only).
    censor : bool, default False
        Run the no-event tokens to the end of follow-up.

    Returns
    -------
    x, a, y, b : torch.Tensor [B, T]
        Input tokens and ages (days), target tokens and ages.
    """
    tokens = pk.T[idx]
    ages = pk.A[idx].clone()
    real = tokens > 0
    minus_inf = torch.full_like(ages, -float('inf'))
    real_ages = torch.where(real, ages, minus_inf)
    # age of the person's last record: no-event tokens stop there, because after it we do not know what happened
    last_age = real_ages.max(1).values
    if censor:
        last_age = torch.maximum(last_age, pk.END[idx])
    if train:
        ages = _shift_lifestyle(tokens, ages, gen, pk.device)
    masked = torch.full_like(ages, MASK_TIME)
    ages = torch.where(real, ages, masked)
    no_event_tok, no_event_age = _no_event_tokens(last_age, tokens.shape[0], gen, pk.device)
    tok = torch.cat([tokens, no_event_tok], 1)
    age = torch.cat([ages, no_event_age], 1)
    tok, age = _sort_by_age(tok, age)
    return _inputs_and_targets(tok, age)


def _shift_lifestyle(tokens, ages, gen, device):
    """Shift the ages of the lifestyle tokens by a random -20 to +40 years, as in Delphi.

    Lifestyle is measured once, at the assessment visit, but stands for habits held for years; the shift stops
    the model from tying lifestyle to the visit age.
    """
    life = (tokens >= 4) & (tokens <= 12)
    if life.any():
        shift = torch.randint(-20 * 365, 40 * 365, ages.shape, generator=gen, device=device)
        shift = shift.float()
        ages = ages + life.float() * shift
    return ages


def _no_event_tokens(last_age, n, gen, device):
    """No-event tokens at 20 uniform random ages in 0-100 years, kept up to each person's last record."""
    K = int(round(100 / NO_EVENT_RATE_YEARS))
    no_event_age = torch.randint(1, 36525, (n, K), generator=gen, device=device)
    no_event_age = no_event_age.float()
    keep = no_event_age <= last_age[:, None]
    no_event = torch.full_like(no_event_age, NO_EVENT, dtype=torch.long)
    nothing = torch.zeros_like(no_event_age, dtype=torch.long)
    no_event_tok = torch.where(keep, no_event, nothing)
    no_event_age = torch.where(keep, no_event_age, torch.full_like(no_event_age, MASK_TIME))
    return no_event_tok, no_event_age


def _sort_by_age(tok, age):
    """Sort each sequence by age (stable, so records on one day keep their order) and cut the common padding."""
    order = torch.sort(age, dim=1, stable=True).indices
    tok = torch.gather(tok, 1, order)
    age = torch.gather(age, 1, order)
    n_pad = (tok == PAD).sum(1)
    lead = int(n_pad.min().item())
    tok = tok[:, lead:]
    age = age[:, lead:]
    # a batch where everyone has one record would leave no target, so we keep at least two positions
    if tok.shape[1] < 2:
        tok = torch.nn.functional.pad(tok, (1, 0), value=PAD)
        age = torch.nn.functional.pad(age, (1, 0), value=MASK_TIME)
    return tok, age


def _inputs_and_targets(tok, age):
    """Inputs are every position but the last, targets every position but the first; padded inputs get no target."""
    x = tok[:, :-1]
    a = age[:, :-1]
    y = tok[:, 1:].clone()
    b = age[:, 1:].clone()
    pad_input = x == PAD
    y[pad_input] = PAD
    b[pad_input] = MASK_TIME
    return x, a, y, b


def iterate(pk, order, batch_size, seed, train=False, censor=False):
    """Deterministic batches over the people in `order`.

    Batch j is drawn with the seed seed * 1000003 + j, so the same people and seed give the same batches whatever
    model is being trained or evaluated, and two models can be compared on identical inputs.

    Parameters
    ----------
    pk : Packed
    order : ndarray of int
        Person indices, in batch order.
    batch_size, seed : int
    train, censor : bool
        Passed to make_batch.

    Yields
    ------
    idx : torch.Tensor
        Person indices of the batch.
    (x, a, y, b) : tuple of torch.Tensor
        The batch (make_batch).
    """
    for j, s in enumerate(range(0, len(order), batch_size)):
        people = order[s:s + batch_size]
        idx = torch.as_tensor(people, device=pk.device)
        batch_seed = int(seed) * 1000003 + j
        g = torch.Generator(device=pk.device)
        g.manual_seed(batch_seed)
        batch = make_batch(pk, idx, g, train=train, censor=censor)
        yield idx, batch
