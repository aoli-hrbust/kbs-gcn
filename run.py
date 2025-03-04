import traceback

from src.models.gcntsne import train_main
from pathlib import Path as P
import logging
import numpy as np
import warnings
import itertools
from src.vis.visualize import *
from src.utils.io_utils import *
from joblib import Parallel, delayed
from tqdm import tqdm
from src.utils.torch_utils import get_device
import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action="ignore")
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    root = P("./json/")
    method = "GTSNE_MvAE"
    dataname = "COIL-20"
    eta = 10
    epochs = 200
    eval_epochs = 10

    savedir = P(f"output/{method}/{dataname}-{eta}")

    datapath = P("./data/").joinpath(dataname)
    train_main(
        datapath=datapath,
        eta=eta / 100,
        device=get_device(),
        savedir=savedir,
        epochs=epochs,
        eval_epochs=eval_epochs,
    )


