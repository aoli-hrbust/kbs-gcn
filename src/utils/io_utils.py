import itertools
import json
import logging
import pickle
import time
from pathlib import Path as P
from pprint import pformat
from typing import Tuple, Union
import jsons
import matplotlib.pyplot as plt
from typing import Literal, List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import subprocess
import os
import sys


def is_file_newer(file_a, file_b):
    # 获取文件a和文件b的状态信息
    stat_a = os.stat(file_a)
    stat_b = os.stat(file_b)

    # 比较文件a和文件b的修改时间
    if stat_a.st_mtime > stat_b.st_mtime:
        return True
    else:
        return False


def read_excel(data_xlsx: Union[str, P]):
    '''
    在读入Excel文件之前，先自动收集csv。这样省去了手动收集的麻烦。
    '''
    data_xlsx = P(data_xlsx)
    cwd = data_xlsx.parent.parent.absolute()
    script = cwd.joinpath('collect_csv.py')
    json_dir = cwd.joinpath('json')
    # project_dir = cwd.parent.parent
    project_dir = os.path.join(__file__, '..', '..', '..')
    project_dir = os.path.abspath(project_dir)
    # print('x', project_dir)
    assert script.exists(), script

    t = time.time()
    need = (not data_xlsx.exists() or any(Parallel(n_jobs=4, backend='threading')(
        delayed(is_file_newer)(a, data_xlsx) for a in json_dir.rglob("*.json")))) and  'previous' not in data_xlsx.name
    t = time.time() - t
    print(f'Checked {data_xlsx} in {t:.2}s, result: {(need)}')

    if (need):
        print(f'AutoRun collect_csv.py')
        env = os.environ.copy()
        env['PYTHONPATH']=str(project_dir)
        subprocess.check_call([sys.executable, 'collect_csv.py'], cwd=cwd, env=env)
    else:
        print('No need to run')

    return pd.read_excel(str(data_xlsx))


def make_directory(*paths):
    for p in paths:
        P(p).mkdir(parents=True, exist_ok=True)


def convert_png2eps(indir: P, outdir: P, type: Literal["jpg", "png"]):
    outdir.mkdir(parents=True, exist_ok=True)
    png_files = list(indir.glob(f"*.{type}"))

    for infile in tqdm(png_files):
        outfile = outdir.joinpath(infile.name.replace(f".{type}", ".eps"))
        cmd = f"bmeps -t {type} -c {infile} {outfile}".split()
        try:
            subprocess.run(cmd, check=True)
        except:
            import traceback

            traceback.print_exc()


def encode_path(**kwargs):
    items = sorted(kwargs.items(), key=lambda x: x[0])
    return "-".join([f"{k}={v}" for k, v in items])


def kv_product(**kwargs):
    """
    >>> for kwargs in kv_product(a='abc', b='xyz'): print(kwargs)
    """
    for val in itertools.product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), val))


def has_var(savedir: P, name: str):
    """
    Judge savedir exists variable.
    """
    for suffix in ".pkl .json .pt .pth".split():
        f = savedir.joinpath(name).with_suffix(suffix)
        if f.exists():
            return True
    return False


def save_var(savedir: P, var, name: str):
    """
    Save a single variable to savedir.
    """
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name).with_suffix(".pkl")
    pickle.dump(var, f.open("wb"))
    logging.info(f"Save Var to {f}")


def save_variables(savedir: P, variables: dict):
    for key, val in variables.items():
        save_var(savedir, val, key)


def load_var(savedir: P, name: str):
    """
    Load variable from savedir.
    """
    f = savedir.joinpath(name).with_suffix(".pkl")
    # assert f.exists(), (savedir, name, f)
    try:
        var = pickle.load(f.open("rb"))
    except FileNotFoundError:
        import torch

        var = torch.load(f.with_suffix(".pt"), "cpu")

    logging.info(f"Load Var from {f}")
    return var


def load_variables(savedir: P, keys: list):
    return {k: load_var(savedir, name=k) for k in keys}


def save_fig(savedir: P, name: str):
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name)
    assert f.suffix, f"save_fig must have suffix, e.g., .png/.eps {name}"
    plt.tight_layout()
    plt.savefig(str(f))
    plt.close()
    logging.info(f"Save Fig to {f}")


def save_json(savedir: P, var, name: str):
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name).with_suffix(".json")
    f.write_text(jsons.dumps(var, jdkwargs=dict(indent=4)))
    logging.info(f"Save Var to {f}")


def load_json(savedir: P, name: str):
    f = savedir.joinpath(name).with_suffix(".json")
    assert f.exists(), (savedir, name, f)
    var = json.load(f.open())
    logging.info(f"Load Var from {f}")
    return var


def train_begin(savedir: P, config: dict, message: str = None):
    message = message or "Train begins\n"
    logging.info(f"{message} {pformat(config)}")
    save_json(savedir, config, "config")


def train_end(savedir: P, metrics: dict, message: str = None):
    message = message or "Train ends"
    logging.info(f"{message} {metrics}")
    save_json(savedir, metrics, "metrics")


def get_all_dataname(datadir: P, exclude=None):
    res = datadir.iterdir()
    res = filter(lambda f: f.suffix == ".mat", res)
    res = map(lambda f: f.name, res)
    if exclude is not None:
        res = filter(lambda f: any(x not in f for x in exclude), res)
    return list(res)
