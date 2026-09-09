import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import pipeline_engine as pe
from project_io import dataset_fingerprint, read_cache


@pytest.mark.parametrize('outcome', ['better', 'worse', 'equal', 'error', 'missing', 'verify_error', 'train_error'])
def test_pipeline_candidate_flow(tmp_path, monkeypatch, outcome):
    for attr, name in [('ROOT', '.'), ('CKPT_DIR', 'production'),
                       ('CANDIDATE_DIR', 'candidates'), ('BACKUP_DIR', 'backups')]:
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        monkeypatch.setattr(pe, attr, path)
    monkeypatch.setattr(pe, 'STATUS_FILE', tmp_path / 'status.json')
    monkeypatch.setattr(pe, 'LOCK_FILE', tmp_path / 'lock.json')
    monkeypatch.setattr(pe.time, 'sleep', lambda _: None)
    data_dir = tmp_path / 'datasets'
    data_dir.mkdir()
    builtin = tmp_path / 'base.csv'
    builtin.write_text('date,value\n2026-01-01,1\n')
    for h in pe.HORIZONS:
        (pe.CKPT_DIR / f'gumnet_h{h}.pt').write_bytes(b'old')

    def frame(error):
        return pd.DataFrame({'Model': ['GUMNet'] * 2, 'Horizon': ['1d', '5d'],
                             'Upload': ['a', 'a'], 'Ngày': pd.to_datetime(['2026-01-02', '2026-01-06']),
                             'Target': ['MG95'] * 2, 'Thực tế': [100.] * 2,
                             '% Lệch': [error] * 2, 'Sai lệch': [error] * 2})

    calls = []

    def simulate(base_path, upload_files, start_date, sel_horizons=None,
                 sel_models=None, log_fn=None, checkpoint_dir=None):
        calls.append(checkpoint_dir)
        current = (pe.CKPT_DIR / 'gumnet_h1.pt').read_bytes()
        if checkpoint_dir is not None:
            assert current == b'old', 'Candidate must be evaluated before promotion'
            if outcome == 'error':
                raise RuntimeError('candidate evaluation failed')
            result = frame(8 if outcome == 'worse' else 6 if outcome == 'equal' else 4)
            return result.iloc[:1] if outcome == 'missing' else result
        if len(calls) > 1 and outcome == 'verify_error':
            raise RuntimeError('verification failed')
        return frame(4 if current == b'new' else 6)

    def train(cmd, **kwargs):
        assert '--update_data' in cmd
        assert Path(cmd[cmd.index('--data-path') + 1]).exists()
        folder = Path(cmd[cmd.index('--output_dir') + 1])
        for h in pe.HORIZONS:
            (folder / f'gumnet_h{h}.pt').write_bytes(b'new')
        return SimpleNamespace(returncode=1 if outcome == 'train_error' else 0)

    worker = SimpleNamespace(BUILTIN_CSV=builtin,
                             load_df=lambda _: pd.DataFrame({'Ngày': pd.to_datetime(['2026-01-01'])}),
                             run_upload_simulation=simulate)
    monkeypatch.setitem(sys.modules, 'backtest_worker', worker)
    monkeypatch.setattr(pe.subprocess, 'run', train)
    monkeypatch.setattr(pe, '_validate_candidate', lambda *args: {h: 0.1 for h in pe.HORIZONS})
    pe.run_pipeline_task('test', 'batch', 10, force_retrain=True)
    status = pe.get_pipeline_status()
    assert not pe.LOCK_FILE.exists()
    assert not status['is_running']
    for h in pe.HORIZONS:
        assert (pe.CKPT_DIR / f'gumnet_h{h}.pt').read_bytes() == (b'new' if outcome == 'better' else b'old')
    fp, cached = read_cache(tmp_path / 'simulation_cache.json')
    assert cached['% Lệch'].mean() == (4 if outcome == 'better' else 6)
    assert pe.already_trained(fp) == (outcome in {'better', 'worse', 'equal'})
    assert status['status'] == ('complete' if outcome in {'better', 'worse', 'equal'} else 'failed')
    if outcome in {'better', 'worse', 'equal'}:
        pe.reset_pipeline_status()
        assert pe.already_trained(fp)
        with patch.object(pe.subprocess, 'Popen') as spawn:
            assert pe.launch_pipeline_background('again', 0, [], True)['reason'] == 'already_trained'
            spawn.assert_not_called()


def test_fingerprint_detects_content_with_preserved_timestamp(tmp_path):
    path = tmp_path / 'a.csv'
    path.write_bytes(b'old')
    before = dataset_fingerprint(tmp_path, pe.HORIZONS)
    stat = path.stat()
    path.write_bytes(b'new')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert dataset_fingerprint(tmp_path, pe.HORIZONS) != before


def test_worker_cache_separates_checkpoint_versions(tmp_path):
    import backtest_worker as bw
    folder = tmp_path / 'candidate'
    folder.mkdir()
    checkpoint = folder / 'gumnet_h1.pt'
    checkpoint.write_bytes(b'old')
    with patch.object(bw, '_MODEL_CACHE', {}), patch.object(bw, '_swap_src'), \
         patch.object(bw.importlib, 'import_module', side_effect=RuntimeError('load attempted')) as loader:
        bw.load_model('GUMNet', 1, folder)
        bw.load_model('GUMNet', 1, folder)
        assert loader.call_count == 1
        checkpoint.write_bytes(b'new-version')
        bw.load_model('GUMNet', 1, folder)
        assert loader.call_count == 2


def test_training_copy_receives_uploaded_prices(tmp_path, monkeypatch):
    import train_all_horizons as training
    original = tmp_path / 'original.csv'
    original.write_text('Ngày,MG95\n2026-01-01,10\n', encoding='utf-8')
    candidate = tmp_path / 'candidate.csv'
    candidate.write_bytes(original.read_bytes())
    uploads = tmp_path / 'datasets'
    uploads.mkdir()
    (uploads / 'new.csv').write_text('Ngày,MG95\n2026-01-01,12\n2026-01-02,14\n', encoding='utf-8')
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(training, 'DATA_PATH', candidate)
    training.update_training_data()
    result = training.read_data()
    assert result['MG95'].tolist() == [12, 14]
    assert pd.read_csv(original)['MG95'].tolist() == [10]


def test_process_probe_does_not_terminate_child():
    import subprocess
    from project_io import process_alive
    child = subprocess.Popen(
        [sys.executable, '-u', '-c', "print('ready', flush=True); input()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    try:
        assert child.stdout.readline().strip() == 'ready'
        for _ in range(20):
            assert process_alive(child.pid)
            assert child.poll() is None
    finally:
        child.communicate('\n', timeout=10)
    assert child.returncode == 0
    assert not process_alive(child.pid)
