"""
Focused tests for postlib.explore.initalize_case, the exploration history loader.

These tests build synthetic ``runs/<case>/`` trees in temporary directories and
check the three correctness properties the loader must guarantee:

  * reward files stored directly under ``history/`` are not dropped;
  * global episode indices are contiguous 1..N (no linspace compression);
  * component CSV episodes are split *before* the completeness check, so a
    partial tail is never spliced onto the next segment.

The file is both pytest-compatible (``def test_*`` + asserts) and runnable
standalone (``python3 test_explore.py``) since pytest is not installed here.
"""

import os
import sys
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from postlib.explore import initalize_case  # noqa: E402

NB = 4  # nb_interactions used throughout the tests


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #
def _write_conf(seg_dir, nb=NB):
    os.makedirs(seg_dir, exist_ok=True)
    conf = {
        'runner': {'nb_interactions': nb},
        'simulation': {'viscosity': 1.0, 'dt': 1.0, 'ndrl': 1},
    }
    with open(os.path.join(seg_dir, 'current_conf.yml'), 'w') as f:
        yaml.safe_dump(conf, f)


def _write_rewlog(seg_dir, idx, length, value=1.0):
    """Write rewlog_<idx>.npz with a 'rew' array of the given length."""
    os.makedirs(seg_dir, exist_ok=True)
    arr = np.full((length,), value, dtype=float) if length > 0 else np.zeros((0,), dtype=float)
    np.savez(os.path.join(seg_dir, f'rewlog_{idx:05d}.npz'), rew=arr)


def _write_agg_csv(seg_dir, timestamp, episodes, values=None):
    """Write rewards_aggregated_<timestamp>.csv.

    episodes : list of (episode_number, n_rows). Steps are written scrambled to
               verify the loader sorts each group by 'step'.
    values   : optional dict {episode_number: (R_tau, R_pw, R_v3)} in raw units.
    """
    os.makedirs(seg_dir, exist_ok=True)
    values = values or {}
    rows = []
    for ep, nrows in episodes:
        rt, rp, rv = values.get(ep, (0.1, -0.2, -0.01))
        steps = list(range(nrows))[::-1]  # reversed → forces a sort-by-step
        for step in steps:
            rows.append({
                'timestamp': timestamp, 'episode': ep, 'step': step,
                'mean_reward': 0.0,
                'mean_R_tau': rt, 'mean_R_pw': rp, 'mean_R_v3': rv,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(seg_dir, f'rewards_aggregated_{timestamp}.csv'), index=False)


def _tmp_run():
    d = tempfile.mkdtemp(prefix='explore_test_')
    return d


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_all_segment_types_and_direct_root():
    """(1) Direct root history + round001 + current_history + train/history.
    Also (2) empty rewlog and (6) contiguous global indices."""
    run = _tmp_run()
    try:
        case = 'case_all'
        cp = os.path.join(run, case)
        hist = os.path.join(cp, 'history')

        # (1) Direct files in history/ + (2) an empty rewlog that must be skipped
        _write_conf(hist)
        _write_rewlog(hist, 1, 0)      # empty → skipped
        _write_rewlog(hist, 2, NB)     # complete → episode 1
        _write_agg_csv(hist, '20260101_000000', [(1, NB)])

        r1 = os.path.join(hist, 'round001')
        _write_conf(r1)
        _write_rewlog(r1, 1, NB)
        _write_agg_csv(r1, '20260102_000000', [(1, NB)])

        ch = os.path.join(hist, 'current_history')
        _write_conf(ch)
        _write_rewlog(ch, 1, NB)
        _write_agg_csv(ch, '20260103_000000', [(1, NB)])

        live = os.path.join(cp, 'train', 'history')
        _write_conf(live)
        _write_rewlog(live, 1, NB)
        _write_agg_csv(live, '20260104_000000', [(1, NB)])

        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any mismatch warning fails the test
            cd = initalize_case(run, [case])
        c = cd[case]

        labels = [s['label'] for s in c['history_segments']]
        assert labels == ['history', 'round001', 'current_history', 'train/history'], labels

        # 4 complete reward episodes: history(1) + round001 + current + live
        assert len(c['reward']) == 4
        assert list(c['episode_idx']) == [1, 2, 3, 4], list(c['episode_idx'])
        # first retained episode is the direct-root rewlog_00002 (not dropped)
        prov = c['episode_provenance']['reward']
        assert prov[0] == {'segment': 'history', 'file': 'rewlog_00002.npz'}, prov[0]
        # component episodes match reward episodes
        assert list(c['comp_episode_idx']) == [1, 2, 3, 4]
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_root_not_a_segment_without_direct_files():
    """The history root is only a segment when it directly holds reward files."""
    run = _tmp_run()
    try:
        case = 'case_noroot'
        hist = os.path.join(run, case, 'history')
        r1 = os.path.join(hist, 'round001')
        _write_conf(r1)
        _write_rewlog(r1, 1, NB)
        _write_agg_csv(r1, '20260101_000000', [(1, NB)])

        cd = initalize_case(run, [case])
        labels = [s['label'] for s in cd[case]['history_segments']]
        assert labels == ['round001'], labels
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_partial_npz_skipped():
    """(3) Partial NPZ episodes (< nb_interactions) are skipped."""
    run = _tmp_run()
    try:
        case = 'case_partial_npz'
        r1 = os.path.join(run, case, 'history', 'round001')
        _write_conf(r1)
        _write_rewlog(r1, 1, 0)        # empty
        _write_rewlog(r1, 2, NB)       # complete → ep 1
        _write_rewlog(r1, 3, NB - 1)   # partial → skipped
        _write_rewlog(r1, 4, NB)       # complete → ep 2
        _write_agg_csv(r1, '20260101_000000', [(1, NB), (2, NB)])

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cd = initalize_case(run, [case])
        c = cd[case]
        assert len(c['reward']) == 2
        assert list(c['episode_idx']) == [1, 2]
        assert [p['file'] for p in c['episode_provenance']['reward']] == \
            ['rewlog_00002.npz', 'rewlog_00004.npz']
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_partial_component_csv_at_segment_end_discarded():
    """(4) A partial component episode at the end of a segment is discarded."""
    run = _tmp_run()
    try:
        case = 'case_partial_csv'
        r1 = os.path.join(run, case, 'history', 'round001')
        _write_conf(r1)
        _write_rewlog(r1, 1, NB)       # 1 complete reward episode
        # ep1 complete (value 0.5), ep2 partial tail (nb-1 rows)
        _write_agg_csv(r1, '20260101_000000',
                       [(1, NB), (2, NB - 1)],
                       values={1: (0.5, -0.5, -0.05), 2: (0.9, -0.9, -0.09)})

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cd = initalize_case(run, [case])
        c = cd[case]
        assert list(c['comp_episode_idx']) == [1], list(c['comp_episode_idx'])
        # _eps computed only from the complete ep1 (0.5 × 100 = 50.0)
        assert np.allclose(c['mean_R_tau_eps'], [50.0]), c['mean_R_tau_eps']
        assert np.allclose(c['mean_R_pw_eps'], [-50.0])
        # raw concatenated array holds exactly one complete episode
        assert c['mean_R_tau'].shape[0] == NB
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_numeric_round_sorting():
    """(5) round2 must be processed before round10 (numeric, not lexical)."""
    run = _tmp_run()
    try:
        case = 'case_numsort'
        hist = os.path.join(run, case, 'history')
        # Distinct reward values per round to identify ordering by provenance.
        for name, idx in [('round2', 2), ('round10', 10), ('round1', 1)]:
            seg = os.path.join(hist, name)
            _write_conf(seg)
            _write_rewlog(seg, 1, NB, value=float(idx))
            _write_agg_csv(seg, f'2026010{idx % 10}_000000', [(1, NB)])

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cd = initalize_case(run, [case])
        c = cd[case]
        seg_order = [p['segment'] for p in c['episode_provenance']['reward']]
        assert seg_order == ['round1', 'round2', 'round10'], seg_order
        # reward values follow the same numeric order (1, 2, 10) × 100
        means = [rm for rm in c['reward_mean']]
        assert np.allclose(means, [100.0, 200.0, 1000.0]), means
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_no_partial_tail_mixing_across_segments():
    """(7) A partial CSV tail is never combined with the next segment's rows."""
    run = _tmp_run()
    try:
        case = 'case_nomix'
        hist = os.path.join(run, case, 'history')

        # Segment A: ep1 complete, ep2 partial (nb-1 rows).
        segA = os.path.join(hist, 'round001')
        _write_conf(segA)
        _write_rewlog(segA, 1, NB)     # 1 complete reward episode
        _write_agg_csv(segA, '20260101_000000',
                       [(1, NB), (2, NB - 1)],
                       values={1: (0.5, -0.5, -0.05), 2: (0.7, -0.7, -0.07)})

        # Segment B: a single dangling row that the old loader would have
        # spliced onto segment A's partial tail to fabricate a 2nd episode.
        segB = os.path.join(hist, 'round002')
        _write_conf(segB)
        _write_agg_csv(segB, '20260102_000000',
                       [(1, 1)], values={1: (0.9, -0.9, -0.09)})

        cd = initalize_case(run, [case])  # counts mismatch → warning expected
        c = cd[case]
        # Only ep1 of segment A survives — no fabricated 2nd episode.
        assert list(c['comp_episode_idx']) == [1], list(c['comp_episode_idx'])
        assert np.allclose(c['mean_R_tau_eps'], [50.0]), c['mean_R_tau_eps']
        assert c['mean_R_tau'].shape[0] == NB
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_warning_on_count_mismatch():
    """(8) A warning is emitted when NPZ and component episode counts differ."""
    run = _tmp_run()
    try:
        case = 'case_mismatch'
        r1 = os.path.join(run, case, 'history', 'round001')
        _write_conf(r1)
        _write_rewlog(r1, 1, NB)   # 2 complete reward episodes
        _write_rewlog(r1, 2, NB)
        _write_agg_csv(r1, '20260101_000000', [(1, NB)])  # only 1 comp episode

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cd = initalize_case(run, [case])
        c = cd[case]
        assert len(c['reward']) == 2
        assert list(c['comp_episode_idx']) == [1]
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any('component episode count' in m and 'reward NPZ episode count' in m
                   for m in msgs), msgs
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_missing_history_raises():
    """A case with no history at all raises FileNotFoundError."""
    run = _tmp_run()
    try:
        os.makedirs(os.path.join(run, 'empty_case'), exist_ok=True)
        try:
            initalize_case(run, ['empty_case'])
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError for missing history")
    finally:
        shutil.rmtree(run, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Standalone runner (no pytest required)
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
