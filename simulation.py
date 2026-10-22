"""A simulated cohort with known disease links, for the walkthrough notebook.

The paper's models were trained on UK Biobank, which cannot be shared, so the notebook works on people simulated
here. Each person has a sex at birth, lifestyle at an assessment visit between 40 and 70 years (smoking, BMI and
alcohol, the lifestyle tokens of Delphi), first occurrences of 29 ICD-10 codes, and death. The three childhood
codes can occur from birth and the others from 18 years. Every code has a yearly rate that changes with age and
sex; the links below multiply the rate of an outcome after its exposure, and some diseases raise the rate of death.
Follow-up runs from birth to death or to 2022 (birth years 1937-1970), as in UK Biobank. Because the links are set
by construction, each analysis of the notebook can check whether a model learned them, and true_log_rate gives the
best score any model could reach.
"""
import numpy as np

from aegis.data import Cohort, Vocab, FEMALE, MALE

YEAR = 365.25

# code: (name, yearly rate at age 50, change of the log rate per 10 years, multiplier for women, for men)
DISEASES = {
    'B01': ('Chickenpox', 0.0002, -1.2, 1.0, 1.0),
    'J03': ('Tonsillitis', 0.0010, -0.6, 1.1, 0.9),
    'J45': ('Asthma', 0.0030, -0.1, 1.2, 0.8),
    'F32': ('Depression', 0.0040, -0.1, 1.4, 0.7),
    'F41': ('Anxiety', 0.0035, -0.2, 1.3, 0.7),
    'F17': ('Tobacco use', 0.0020, -0.2, 0.9, 1.1),
    'F10': ('Alcohol use disorder', 0.0010, -0.1, 0.5, 1.5),
    'E66': ('Obesity', 0.0040, 0.1, 1.1, 0.9),
    'E78': ('Hypercholesterolaemia', 0.0080, 0.4, 1.0, 1.0),
    'I10': ('Hypertension', 0.0120, 0.5, 1.0, 1.1),
    'E11': ('Type 2 diabetes', 0.0040, 0.4, 0.8, 1.2),
    'K21': ('Reflux disease', 0.0050, 0.2, 1.0, 1.0),
    'M17': ('Knee osteoarthritis', 0.0030, 0.6, 1.2, 0.8),
    'G47': ('Sleep disorder', 0.0030, 0.2, 1.0, 1.0),
    'I25': ('Chronic ischaemic heart disease', 0.0025, 0.6, 0.6, 1.4),
    'I21': ('Myocardial infarction', 0.0015, 0.6, 0.5, 1.5),
    'I48': ('Atrial fibrillation', 0.0015, 0.9, 0.8, 1.2),
    'I50': ('Heart failure', 0.0008, 0.9, 1.0, 1.0),
    'I63': ('Stroke', 0.0008, 0.8, 1.0, 1.0),
    'N18': ('Chronic kidney disease', 0.0010, 0.8, 1.0, 1.0),
    'J44': ('COPD', 0.0010, 0.6, 1.0, 1.0),
    'K70': ('Alcoholic liver disease', 0.0003, 0.2, 1.0, 1.0),
    'H25': ('Cataract', 0.0030, 1.0, 1.0, 1.0),
    'K57': ('Diverticular disease', 0.0030, 0.6, 1.0, 1.0),
    'C34': ('Lung cancer', 0.0003, 0.7, 1.0, 1.0),
    'C50': ('Breast cancer', 0.0020, 0.3, 1.0, 0.0),
    'C61': ('Prostate cancer', 0.0020, 0.8, 0.0, 1.0),
    'C18': ('Colon cancer', 0.0005, 0.6, 1.0, 1.0),
    'G30': ('Alzheimer disease', 0.0001, 1.5, 1.1, 0.9),
}
# The codes a child can get; every other code starts at ADULT_AGE. Without this limit the rates of tobacco use,
# alcohol use disorder, depression and anxiety would be highest at birth (their log rates fall with age), so the
# simulation gave toddlers these records and put the exposures decades before their outcomes.
CHILDHOOD = ('B01', 'J03', 'J45')
ADULT_AGE = 18.0
# exposure -> outcome: rate ratio after the exposure (the known structure the model should learn)
LINKS = {('E66', 'E11'): 3.0, ('E66', 'I10'): 2.0, ('E66', 'M17'): 2.0, ('E66', 'G47'): 2.5,
         ('I10', 'I21'): 2.0, ('I10', 'I63'): 2.5, ('I10', 'N18'): 2.0, ('I10', 'I50'): 2.0,
         ('E11', 'I21'): 2.0, ('E11', 'N18'): 3.0, ('E11', 'H25'): 1.5, ('E78', 'I21'): 2.0, ('E78', 'I25'): 2.0,
         ('I21', 'I50'): 4.0, ('I21', 'I25'): 3.0, ('I48', 'I63'): 4.0, ('I48', 'I50'): 2.5,
         ('F17', 'C34'): 8.0, ('F17', 'J44'): 6.0, ('F17', 'I21'): 1.8, ('F10', 'K70'): 15.0, ('F10', 'F32'): 2.0,
         ('F32', 'F41'): 3.0, ('F41', 'F32'): 3.0, ('J45', 'J44'): 2.0}
DEATH_LINKS = {'C34': 15.0, 'I50': 4.0, 'G30': 5.0, 'J44': 2.5, 'N18': 2.0, 'I21': 2.0, 'K70': 4.0}
# lifestyle at the assessment visit (tokens 4-12): share of people, and which rates it multiplies
LIFESTYLE = {'smoking': ((7, 8, 9), (0.75, 0.10, 0.15), {'F17': 8.0, 'C34': 3.0, 'J44': 2.0}),
             'bmi': ((4, 5, 6), (0.30, 0.45, 0.25), {'E66': 6.0, 'E11': 1.5}),
             'alcohol': ((10, 11, 12), (0.50, 0.35, 0.15), {'F10': 5.0, 'K70': 2.0})}


def simulate(n, seed=0, step=0.25, max_age=85.0):
    """Simulate a cohort of first occurrences and deaths whose disease links are known by construction.

    Every person has a sex, lifestyle at an assessment visit (age 40-70), first occurrences of the diseases above
    (the CHILDHOOD codes from birth, the others from ADULT_AGE), and death; follow-up ends at death, at max_age or
    in 2022 (birth years 1937-1970). Rates are held constant within each time step, so a link acts from the step
    after its exposure.

    Parameters
    ----------
    n : int
        Number of people.
    seed : int, default=0
        Seed of the one generator behind every draw, so a seed gives the same cohort.
    step : float, default=0.25
        Time step of the simulation (years).
    max_age : float, default=85.0
        Age (years) at which follow-up stops for everyone still alive.

    Returns
    -------
    coh : Cohort
        One record per (person, token, age), ages in days, with `end` set to the end of follow-up.
    codes : list of str
        The ICD-10 codes of the simulation, in the order of DISEASES.
    """
    V = Vocab()
    rng = np.random.default_rng(seed)
    codes = list(DISEASES)
    tok = np.array([V.stoi[c] for c in codes])
    col = {c: j for j, c in enumerate(codes)}
    base, slope, sex_mult = _rate_table(codes)

    # We keep the order of the draws (sex, birth year, visit, lifestyle, age steps) so that a seed reproduces the
    # cohort. 46% men is about the share in UK Biobank.
    male = rng.random(n) < 0.46
    # end of follow-up in 2022 as an age; the uniform fraction spreads the birthdays over the year
    censor = 2022 - rng.integers(1937, 1971, n) + rng.random(n)
    visit = rng.uniform(40, 70, n)
    life, life_mult = _lifestyle(rng, n, col, len(codes))
    adult = np.array([c not in CHILDHOOD for c in codes])

    onset = np.full((n, len(codes)), np.inf)  # age in years, inf while the disease has not occurred
    death = np.full(n, np.inf)
    for t in np.arange(0, max_age, step):
        alive = (t < censor) & ~np.isfinite(death)
        if not alive.any():
            break
        h = _disease_rates(t, base, slope, sex_mult, male, life_mult)
        if t < ADULT_AGE:
            # ADULT_AGE is a multiple of the step, so no adult code starts before it; we still draw the uniform
            # numbers below for every code, so the draws of later steps keep their order
            h[:, adult] = 0.0
        # `had` is taken at the start of the step, so two diseases drawn in one step do not raise each other
        had = np.isfinite(onset)
        _apply_links(h, had, col)
        # With the rate constant over the step, a first occurrence happens with probability 1 - exp(-h * step), at
        # a uniform time within the step.
        new = alive[:, None] & ~had & (rng.random(h.shape) < 1 - np.exp(-h * step))
        onset[new] = t + rng.random(new.sum()) * step
        hd = _death_rate(t, had, col)
        dies = alive & (rng.random(n) < 1 - np.exp(-hd * step))
        # A disease drawn in the step of death may fall after the death age; the `<= end` test in _records drops it.
        death[dies] = t + rng.random(dies.sum()) * step

    end = np.minimum(np.minimum(death, censor), max_age)
    P, K, A = _records(n, V, tok, male, visit, life, onset, death, end)
    coh = Cohort.from_long(np.array(P), np.array(K, np.int32), np.array(A, np.float32) * YEAR)
    coh.end = (end * YEAR).astype(np.float32)
    return coh, codes


def _rate_table(codes):
    # yearly rate at 50, change per 10 years, and multiplier for women (row 0) and men (row 1)
    base = np.array([DISEASES[c][1] for c in codes])
    slope = np.array([DISEASES[c][2] for c in codes])
    sex_mult = np.array([[DISEASES[c][3] for c in codes], [DISEASES[c][4] for c in codes]])
    return base, slope, sex_mult


def _lifestyle(rng, n, col, n_codes):
    # Lifestyle category of each person (0, 1, 2). The highest category multiplies some rates from birth because
    # the habits stand for years before the visit that records them.
    life = {}
    for k, (_, shares, _) in LIFESTYLE.items():
        life[k] = rng.choice(len(shares), n, p=shares)
    life_mult = np.ones((n, n_codes))
    for k, (_, _, effects) in LIFESTYLE.items():
        for c, m in effects.items():
            life_mult[:, col[c]] *= np.where(life[k] == 2, m, 1.0)
    return life, life_mult


def _disease_rates(t, base, slope, sex_mult, male, life_mult):
    return base * np.exp(slope * (t - 50) / 10) * sex_mult[male.astype(int)] * life_mult


def _apply_links(h, had, col):
    # in place, on the rates of the outcomes
    for (e, o), rr in LINKS.items():
        h[:, col[o]] *= np.where(had[:, col[e]], rr, 1.0)


def _death_rate(t, had, col):
    # Gompertz: the rate at 50 grows by 9% a year, multiplied by the diseases that raise it.
    death_mult = np.prod([np.where(had[:, col[c]], m, 1.0) for c, m in DEATH_LINKS.items()], 0)
    return 0.003 * np.exp(0.087 * (t - 50)) * death_mult


def _records(n, V, tok, male, visit, life, onset, death, end):
    # one record per (person, token, age), as in UK Biobank's first occurrences
    P, K, A = [], [], []
    for i in range(n):
        P.append(i)
        K.append(MALE if male[i] else FEMALE)
        A.append(0.0)
        # People who died or left before their visit age never attended, so they have no lifestyle record.
        if visit[i] < end[i]:
            for k, (toks, _, _) in LIFESTYLE.items():
                P.append(i)
                K.append(toks[life[k][i]])
                A.append(visit[i])
        for j in np.flatnonzero(onset[i] <= end[i]):
            P.append(i)
            K.append(tok[j])
            A.append(onset[i, j])
        if death[i] <= end[i]:
            P.append(i)
            K.append(V.death)
            A.append(death[i])
    return P, K, A


def caused_codes():
    """Codes whose rate depends on more than age and sex in the simulation.

    These are the outcomes of a link in LINKS and the codes raised by the highest lifestyle category. A model can
    beat an age- and sex-only score on them. On the other codes the true rate depends on age and sex alone, so its
    stratified AUC is close to 0.5 (a little above, from the age trend within a 5-year bracket), which makes them a
    check of the evaluation.

    Returns
    -------
    set of str
        ICD-10 codes with a cause beyond age and sex.
    """
    caused = set()
    for exposure, outcome in LINKS:
        caused.add(outcome)
    for toks, shares, effects in LIFESTYLE.values():
        for c in effects:
            caused.add(c)
    return caused


def true_log_rate(tokens, ages, k, age, vocab):
    """The simulation's own log-rate of token k for one person at an age, from what their history shows before it.

    The simulation draws every event from this rate, so its AUC on the same cases and controls is the reference a
    model is measured against. It is not a strict ceiling: the lifestyle effect counts only once the visit record is
    in the history, as a model would see it, although the simulation applies it from birth, so a model that infers
    lifestyle from earlier diagnoses (tobacco use from F17, say) can score slightly better before the visit.
    true_log_rates computes the same rate for many people at once.

    Parameters
    ----------
    tokens, ages : list
        The person's history (ages in days).
    k : int
        Token of the disease, or death.
    age : float
        Age in years at which the rate is taken.
    vocab : aegis.data.Vocab

    Returns
    -------
    float
        Log of the yearly rate (nan for a token outside the simulation, -inf for a code impossible for the sex).

    Notes
    -----
    We take the rate at the exact age, while the simulation holds it constant within each step of 0.25 years;
    the two differ by at most slope * step / 10 on the log scale.
    """
    seen = set()
    for t, g in zip(tokens, ages):
        if g < age * YEAR:
            seen.add(t)
    if k == vocab.death:
        return _death_log_rate(age, seen, vocab)

    c = vocab.itos[k]
    if c not in DISEASES:
        return np.nan
    name, base, slope, female_mult, male_mult = DISEASES[c]
    # the sex token is at age 0, so it is always in the history
    if MALE in tokens:
        sex_mult = male_mult
    else:
        sex_mult = female_mult
    # breast cancer in men and prostate cancer in women: rate 0, so we return -inf rather than log(0)
    if sex_mult == 0:
        return -np.inf
    # an adult code has rate 0 before ADULT_AGE, as in simulate
    if c not in CHILDHOOD and age < ADULT_AGE:
        return -np.inf
    log_rate = np.log(base) + slope * (age - 50) / 10 + np.log(sex_mult)
    return _add_exposures(log_rate, c, seen, vocab)


def true_log_rates(coh, people, ages_years, k, vocab, first=None):
    """true_log_rate of token k for many (person, age) pairs at once, with the same arithmetic.

    The notebook scores every case and control of a disease with the true rate; one Python call per record would
    take minutes on 500,000 people, so we compute the rate from each person's first age of every token instead.
    The operations and their order follow true_log_rate exactly, so both give the same floats.

    Parameters
    ----------
    coh : aegis.data.Cohort
        The people, with one record per token (first occurrences).
    people : (m,) array_like of int
        Index of the person of each record to score.
    ages_years : (m,) array_like of float32
        Age in years of each record to score.
    k : int
        Token of the disease, or death.
    vocab : aegis.data.Vocab
    first : dict, optional
        first_ages(coh), when the caller scores many tokens of the same cohort and computes it once.

    Returns
    -------
    (m,) ndarray of float64
        Log of the yearly rate (nan for a token outside the simulation, -inf for a code impossible for the sex).
    """
    people = np.asarray(people, np.int64)
    ages = np.asarray(ages_years, np.float32)
    assert people.shape == ages.shape, 'one age per scored record'
    # true_log_rate compares the float32 record ages with age * YEAR in double precision
    cut = ages.astype(np.float64) * YEAR
    if first is None:
        first = first_ages(coh)

    def seen(t):
        return first.get(t, np.full(len(coh), np.inf))[people] < cut

    # under NumPy 1.x a float32 scalar minus an int is a double, so true_log_rate works in double from here on
    since_50 = ages.astype(np.float64) - 50
    if k == vocab.death:
        log_rate = np.log(0.003) + 0.087 * since_50
        for c, m in DEATH_LINKS.items():
            log_rate[seen(vocab.stoi[c])] += np.log(m)
        return log_rate
    c = vocab.itos[k]
    if c not in DISEASES:
        return np.full(len(people), np.nan)
    name, base, slope, female_mult, male_mult = DISEASES[c]
    male = np.isfinite(first.get(MALE, np.full(len(coh), np.inf)))[people]
    log_rate = np.full(len(people), -np.inf)
    for is_male, mult in ((False, female_mult), (True, male_mult)):
        # a rate of 0 (breast cancer in men, prostate cancer in women) stays -inf, as in true_log_rate
        if mult == 0:
            continue
        rows = male == is_male
        log_rate[rows] = np.log(base) + slope * since_50[rows] / 10 + np.log(mult)
    for toks, shares, effects in LIFESTYLE.values():
        if c in effects:
            log_rate[seen(toks[2])] += np.log(effects[c])
    for (e, o), rr in LINKS.items():
        if o == c:
            log_rate[seen(vocab.stoi[e])] += np.log(rr)
    # last, so that the sums above stay those of true_log_rate; -inf plus a finite number stays -inf anyway
    if c not in CHILDHOOD:
        log_rate[ages.astype(np.float64) < ADULT_AGE] = -np.inf
    return log_rate


def first_ages(coh):
    """First age (days) of every token per person, as a dict from token to an [n_people] array (inf if absent)."""
    rows = coh.rows()
    first = {}
    for t in np.unique(coh.tok):
        a = np.full(len(coh), np.inf)
        has_t = coh.tok == t
        np.minimum.at(a, rows[has_t], coh.age[has_t].astype(np.float64))
        first[int(t)] = a
    return first


def _death_log_rate(age, seen, vocab):
    log_rate = np.log(0.003) + 0.087 * (age - 50)
    for c, m in DEATH_LINKS.items():
        if vocab.stoi[c] in seen:
            log_rate += np.log(m)
    return log_rate


def _add_exposures(log_rate, c, seen, vocab):
    # lifestyle first, then the links, in a fixed order so that the float sum is reproducible
    for toks, shares, effects in LIFESTYLE.values():
        if c in effects and toks[2] in seen:
            log_rate += np.log(effects[c])
    for (e, o), rr in LINKS.items():
        if o == c and vocab.stoi[e] in seen:
            log_rate += np.log(rr)
    return log_rate
