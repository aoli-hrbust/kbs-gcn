# MIT License

# Copyright (c) 2023 Ao Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import math

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.data import MultiviewDataset, make_mask
from src.utils.io_utils import train_begin, save_var, train_end
from src.utils.metrics import (
    KMeans_Evaluate,
    mse_missing_part,
    MaxMetrics,
)
from src.utils.torch_utils import convert_numpy, torch
from src.vis.visualize import visualize_completion
from .loss import *
from .model import *
from pathlib import Path as P
import random
from easydict import EasyDict

from .ptsne_training import calculate_optimized_p_cond, make_joint


class CompletionTrainer:
    def __init__(
        self,
        lamda: float,
        before: bool,
        savedir: P,
    ) -> None:
        self.loss = MyLoss(lamda, before)
        self.history = []
        self.savedir = savedir

    @torch.no_grad()
    def inference(self,
                  datapath: P,
                  views,
                  eta: float,
                  hidden_dims: int,
                  use_mlp: bool,
                  perplexity: int,
                  device: str,
                  **kwargs,
                  ):
        inputs = self.preprocess(
            datapath=datapath,
            views=views,
            eta=eta,
            device=device,
            perplexity=perplexity,
            save_vars=False,
        )

        model = GCN_IMC_Model(
            hidden_dims=hidden_dims,
            use_mlp=use_mlp,
            perplexity=perplexity,
            in_channels=inputs["data"].view_dims,
        ).to(device)
        model.load_state_dict(torch.load(self.savedir.joinpath('best_model.pth')))

        with torch.no_grad():
            x = model(inputs)
        mse = mse_missing_part(X_hat=x["X_hat"], X=x["X_gt"], M=~x["M"])
        metrics = KMeans_Evaluate(x["H_common"], x["data"])
        metrics['MSE'] = mse
        return metrics

    def train(
        self,
        datapath: P,
        views,
        eta: float,
        hidden_dims: int,
        use_mlp: bool,
        perplexity: int,
        device: str,
        eval_epochs: int,
        epochs: int,
        lr: float,
        save_vars: bool,
        save_history: bool,
        save_model: bool,
        **kwargs,
    ):
        inputs = self.preprocess(
            datapath=datapath,
            views=views,
            eta=eta,
            perplexity=perplexity,
            device=device,
            save_vars=save_vars,
        )
        mm = inputs["mm"]

        model = GCN_IMC_Model(
            hidden_dims=hidden_dims,
            use_mlp=use_mlp,
            perplexity=perplexity,
            in_channels=inputs["data"].view_dims,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_mse = 999
        best_acc = -1
        x = inputs.copy()

        for epoch in range(epochs):
            model.train()
            x = model(x)
            loss = self.loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1

            if (1 + epoch) % eval_epochs == 0:
                model.eval()
                with torch.no_grad():
                    x = model(x)
                mse = mse_missing_part(X_hat=x["X_hat"], X=x["X_gt"], M=~x["M"])
                cluster_metrics = KMeans_Evaluate(x["H_common"], x["data"])
                mm.update(MSE=mse, **cluster_metrics)
                logging.info(
                    f"epoch {epoch:04} loss {loss.item():.4f} {mm.report(current=True)}"
                )
                if mse < best_mse or cluster_metrics["ACC"] > best_acc:
                    best_mse = mse
                    inputs.update(x)
                if save_history:
                    his = dict(loss=convert_numpy(loss), mse=mse, **cluster_metrics)
                    self.history.append(his)
                if save_model:
                    torch.save(model.state_dict(), self.savedir.joinpath("best_model.pth"))

        inputs["history"] = self.history
        return self.postprocess(inputs, save_vars)

    def postprocess(self, inputs: dict, save_vars: bool):
        mm: MaxMetrics = inputs["mm"]
        metrics = mm.report(current=False)
        if save_vars:
            for name in "history H_common X_hat X_gt".split():
                save_var(self.savedir, convert_numpy(inputs[name]), name)

        return metrics

    def preprocess(
        self,
        datapath: P,
        views,
        eta: float,
        perplexity: int,
        device: str,
        save_vars: bool,
    ):
        paired_rate: float = 1 - eta

        data = MultiviewDataset(
            datapath=datapath,
            view_ids=views,
        )
        logging.info(
            "Loaded dataset {}, #views {} paired_rate {}".format(
                data.name,
                data.viewNum,
                paired_rate,
            )
        )

        M = make_mask(
            paired_rate=paired_rate,
            sampleNum=data.sampleNum,
            viewNum=data.viewNum,
            kind="partial",
        )

        X_view = [data.X[v][M[:, v]] for v in range(data.viewNum)]
        scaler_view = [MinMaxScaler() for _ in range(data.viewNum)]
        for v in range(data.viewNum):
            X_view[v] = scaler_view[v].fit_transform(X_view[v])
        X_view = convert_tensor(X_view, torch.float, device)

        X_gt = [None] * data.viewNum
        scaler_view = [MinMaxScaler() for _ in range(data.viewNum)]
        for v in range(data.viewNum):
            X_gt[v] = scaler_view[v].fit_transform(data.X[v])
        X_gt = convert_tensor(X_gt, torch.float, device)

        S_view = [
            calculate_optimized_p_cond(x, math.log2(perplexity), dev=device)
            for x in X_view
        ]

        P_view = [make_joint(s) for s in S_view]

        res = dict(
            data=data,
            viewNum=data.viewNum,
            M=convert_tensor(M, torch.bool, device),
            S_view=S_view,
            P_view=P_view,
            X_view=X_view,
            X_gt=X_gt,
            mm=MaxMetrics(MSE=False, ACC=True, NMI=True, PUR=True, F1=True),
        )
        if save_vars:
            savedir = self.savedir
            save_var(savedir, convert_numpy(S_view), "S_view")
            save_var(savedir, convert_numpy(P_view), "P_view")
            save_var(savedir, convert_numpy(X_view), "X_view")
            save_var(savedir, convert_numpy(M), "M")

        return res
