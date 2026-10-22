"""Demo of the AEGIS core code on random data (no UK Biobank data; about 1-5 minutes on a CPU, no GPU needed).

Usage, from this folder:  python demo.py

It trains tiny models (non-private and DP-SGD), runs the privacy audit, samples synthetic people, runs the copying
test, the DCR check and rate calibration, and checks properties that must hold whatever the data. The numbers are
toy-scale; they are not the results of the paper. Python puts the folder of this file on the import path, so the
`aegis` folder next to it is found without installing anything.
"""
import sys
import time

import numpy as np
import torch

from aegis.data import Cohort, Packed, Vocab, iterate, make_planted, make_splits, planted_pool
from aegis.train import train
from aegis.audit import person_groups, run_audit
from aegis.synthetic import ModelGenerator, calibrate, copying_test, distance_to_closest_record, dp_real_rate

# one thread, so that the demo gives the same numbers on every run of a machine
torch.set_num_threads(1)
V = Vocab()
# 50 A/B codes and 10 F codes, so that the psychiatric attack has targets
CODES = np.r_[np.array(V.code_tokens())[:50], V.psychiatric()[:10]]
# float32 sums over a batch and over one person differ in the last bits only
LOSS_ATOL = 1e-5
# the logits are float32, so the difference after adding the constant equals it only up to rounding
SHIFT_ATOL = 1e-4
# the kept log-scale is one of the evaluated ones, so it matches to double precision
SCALE_ATOL = 1e-9
# ages are float32 days, so the end of follow-up may exceed 85 years by a rounding error
AGE_ATOL = 1e-3
CFG = dict(arch='tiny', max_events=24, epochs=3, batch=64, lr=3e-3, warmup=10, weight_decay=0.01, patience=5,
           dropout=0.0, tie=True, hier=True, loss_unit='person', init_rate='public', dp_epochs=3, dp_batch=256,
           phys_batch=128, dp_lr=0.01, clip=1.0, delta=1e-5)
checks = []


def check(name, ok):
    """Record and print one check."""
    checks.append((name, bool(ok)))
    print('  [%s] %s' % ('PASS' if ok else 'FAIL', name), flush=True)


def random_cohort(n, seed):
    """n random people: a sex token, 2-7 different codes at sorted ages of 30-75 years, and death for a fifth."""
    rng = np.random.default_rng(seed)
    P, K, A = [], [], []
    for i in range(n):
        codes = rng.choice(CODES, rng.integers(2, 8), replace=False)
        ages = np.sort(rng.uniform(30, 75, len(codes))) * 365.25
        toks, days = [rng.choice([2, 3])] + list(codes), [0.0] + list(ages)
        if rng.random() < 0.2:
            toks, days = toks + [V.death], days + [80 * 365.25]
        P += [i] * len(toks)
        K += toks
        A += days
    return Cohort.from_long(np.array(P), np.array(K, np.int32), np.array(A, np.float32))


def check_data():
    """Random cohorts, splits, planted patients and batches."""
    print('data')
    members, val, nonmembers, ref_data = random_cohort(600, 1), random_cohort(100, 2), random_cohort(300, 3), random_cohort(400, 4)
    splits = make_splits(1000, seed=0)
    # every person in exactly one set: the sets together have 1,000 distinct people
    everyone = np.concatenate(list(splits.values()))
    check('splits are by person, disjoint and complete', len(everyone) == 1000 and len(np.unique(everyone)) == 1000)
    check('vocabulary is fixed: 1,714 tokens', V.size == 1714)
    pool = planted_pool(members.token_counts(V.size), 'rare', 6)
    planted = {k: make_planted(pool, 30, 6, np.random.default_rng(i)) for i, k in enumerate(['in1', 'in8', 'out'])}
    check('planted patients: sex + 6 different codes each', all((planted[k].lengths() == 7).all() for k in planted))
    pk = Packed(members, 24, 'cpu')
    # two models compared on the same people must see the same no-event tokens and lifestyle shifts
    b1 = [t for _, batch in iterate(pk, np.arange(64), 32, seed=3, train=True) for t in batch]
    b2 = [t for _, batch in iterate(pk, np.arange(64), 32, seed=3, train=True) for t in batch]
    check('batches are a deterministic function of (people, seed)', all(torch.equal(x, y) for x, y in zip(b1, b2)))
    return members, val, nonmembers, ref_data, planted


def check_training(members, val, planted):
    """Train a non-private and a DP-SGD model (epsilon 8) on the members and the planted patients."""
    print('training')
    training_data = Cohort.concat([members, planted['in1'], planted['in8'].repeat(8)])
    models = {}
    for eps in (None, 8):
        m, extra = train(training_data, val, V, CFG, 'cpu', epsilon=eps, seed=0, label='dp8' if eps else 'non-private')
        models[eps] = (m, extra)
        check('%s: validation loss is finite' % extra['label'], all(np.isfinite(h['val']) for h in extra['history']))
    privacy = models[8][1]['privacy']
    check('DP-SGD: epsilon spent <= target and close to it (%.2f)' % privacy['epsilon'], 7.0 < privacy['epsilon'] <= 8.0)
    check('DP-SGD: noise was added (noise multiplier %.2f > 0)' % privacy['noise_multiplier'],
          privacy['noise_multiplier'] > 0)
    return models


def check_person_loss(m, members):
    """Per-person clipping needs each person's loss to depend on that person only, not on the batch."""
    x, a, y, b = next(iterate(Packed(members, 24, 'cpu'), np.arange(16), 16, seed=1))[1]
    with torch.no_grad():
        whole = m.person_loss(x, a, y, b)[0]
        alone = torch.stack([m.person_loss(x[i:i + 1], a[i:i + 1], y[i:i + 1], b[i:i + 1])[0][0] for i in range(16)])
    check("per-person loss: a person's loss does not depend on the others in the batch",
          torch.allclose(whole, alone, atol=LOSS_ATOL))


def check_audit(models, members, val, nonmembers, ref_data, planted):
    """Run every attack of the audit on both models with two reference models."""
    print('privacy audit')
    refs = []
    for h in range(2):
        ref_people = ref_data.subset(np.arange(h * 200, h * 200 + 200))
        refs.append(train(ref_people, val, V, dict(CFG, epochs=1), 'cpu', seed=10 + h, label='reference %d' % h)[0])
    msub = members.subset(np.arange(300))
    packs = {k: Packed(c, 24, 'cpu') for k, c in [('members', msub), ('nonmembers', nonmembers),
                                                   ('canary_out', planted['out']), ('canary_in1', planted['in1']),
                                                   ('canary_in8', planted['in8'])]}
    counts = members.token_counts(V.size)
    groups = {'members': person_groups(msub, V, counts), 'nonmembers': person_groups(nonmembers, V, counts)}
    # with reference models and groups every part of the audit must be there
    expected = {'canary', 'loss_attack', 'reference_attack', 'canary_reference', 'membership_advantage',
                'psychiatric_leakage', 'empirical_epsilon_lower_bound', 'one_run_epsilon'}
    for eps in (None, 8):
        res = run_audit(models[eps][0], refs, packs, V, 1e-5, groups=groups)
        aucs = [res['loss_attack']['auc'], res['reference_attack']['auc'], res['psychiatric_leakage']['auc']]
        aucs += [v['auc'] for v in res['canary'].values()]
        name = 'dp8' if eps else 'non-private'
        check('%s: every attack ran' % name, expected <= set(res))
        check('%s: attack AUCs lie in [0, 1]' % name, all(0 <= v <= 1 for v in aucs))
        check('%s: one-run lower bound on epsilon >= 0 (%.2f)' % (name, res['one_run_epsilon']['epsilon']),
              res['one_run_epsilon']['epsilon'] >= 0)


def check_synthetic(m, members, nonmembers, planted):
    """Sample 300 synthetic people twice and run the copying test and the DCR check."""
    print('synthetic cohorts')
    gen = ModelGenerator(m, 'cpu', 128)
    S = gen.sample(V, 300, seed=0)
    # a disease occurs once and time only moves forward, whatever the model
    ages_sorted = all(np.all(np.diff(S.age[S.off[i]:S.off[i + 1]]) >= 0) for i in range(len(S)))
    no_repeats = all(len(set(t)) == len(t) for t in (S.tok[S.off[i]:S.off[i + 1]] for i in range(len(S))))
    check('300 synthetic people, ages in order, no disease twice', len(S) == 300 and ages_sorted and no_repeats)
    check('synthetic follow-up ends by 85 years', float(S.end.max()) <= 85 * 365.25 + AGE_ATOL)
    S2 = gen.sample(V, 300, seed=0)
    check('sampling is reproducible with the same seed', np.array_equal(S.tok, S2.tok) and np.array_equal(S.age, S2.age))
    cp = copying_test({'1': planted['in1'], '8': planted['in8']}, planted['out'], S, V)
    check('copying test: AUCs in [0, 1]', all(0 <= v['auc'] <= 1 for v in cp.values()))
    dcr = distance_to_closest_record(S, members, nonmembers, V, n_synth=200, n_ref=200)
    check('DCR check: share closer to a member in [0, 1] (%.2f)' % dcr['share_closer_to_members'],
          0 <= dcr['share_closer_to_members'] <= 1)


def check_calibration(m, members):
    """Rate calibration must add one constant to every logit, so that rankings and AUCs do not change."""
    print('rate calibration')
    x, a, y, b = next(iterate(Packed(members, 24, 'cpu'), np.arange(16), 16, seed=1))[1]
    with torch.no_grad():
        before = m.logits(x, a, b)
        real_rate = dp_real_rate(members, V, eps=0.01, seed=0)['rate']
        hist = calibrate(m, V, real_rate, 'cpu', n=300, iters=3, batch=128, seed=0, log=lambda *args: None)
        after = m.logits(x, a, b)
    predicted = ~m.not_predicted
    shift = (after - before)[..., predicted]
    check('calibration ran and kept one evaluated scale',
          any(abs(h['log_scale'] - m.c.log_scale) < SCALE_ATOL for h in hist))
    check('calibration adds one constant to every logit (rankings and AUCs unchanged)',
          torch.allclose(shift, torch.full_like(shift, float(m.c.log_scale)), atol=SHIFT_ATOL))


t0 = time.time()
members, val, nonmembers, ref_data, planted = check_data()
models = check_training(members, val, planted)
m = models[8][0].eval()
check_person_loss(m, members)
check_audit(models, members, val, nonmembers, ref_data, planted)
check_synthetic(m, members, nonmembers, planted)
check_calibration(m, members)

ok = all(p for _, p in checks)
print('\n%s: %d of %d checks passed in %.1f min' % ('PASSED' if ok else 'FAILED', sum(p for _, p in checks), len(checks),
                                                    (time.time() - t0) / 60))
sys.exit(0 if ok else 1)
