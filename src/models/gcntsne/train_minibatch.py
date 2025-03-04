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
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.data import MultiviewDataset, make_mask
from src.utils.io_utils import train_begin, save_var, train_end
from src.utils.metrics import (
    KMeans_Evaluate,
    mse_missing_part,
    MaxMetrics,
)
from src.utils.torch_utils import convert_numpy, convert_cpu
from src.data.dataloader import *

from .loss import *
from .model import *
from pathlib import Path as P
from easydict import EasyDict

from .ptsne_training import calculate_optimized_p_cond, make_joint


class MinibatchCompletionTrainer:
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
                  batch_size: int,
                  use_mlp: bool,
                  perplexity: int,
                  device: str,
                  ):
        inputs = self.preprocess(
            datapath=datapath,
            views=views,
            eta=eta,
            batch_size=batch_size,
            shuffle=False,
        )
        self.data = inputs['data']

        self.model = GCN_IMC_Model(
            hidden_dims=hidden_dims,
            use_mlp=use_mlp,
            perplexity=perplexity,
            in_channels=inputs["data"].view_dims,
        ).to(device)
        self.model.load_state_dict(torch.load(self.savedir.joinpath('best_model.pth')))

        metrics, outputs = self.evaluate(
            dataloader=inputs['test_dataloader'],
            device=device,
            inputs=inputs,
            ppl=perplexity,
        )
        return metrics

    def train(
        self,
        datapath: P,
        views,
        eta: float,
        hidden_dims: int,
        batch_size: int,
        shuffle: bool,
        use_mlp: bool,
        perplexity: int,
        device: str,
        eval_epochs: int,
        epochs: int,
        lr: float,
        save_vars: bool,
        save_history: bool,
        save_model: bool,
    ):
        inputs = self.preprocess(
            datapath=datapath,
            views=views,
            eta=eta,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        mm = inputs["mm"]
        self.data = inputs['data']

        self.model = GCN_IMC_Model(
            hidden_dims=hidden_dims,
            use_mlp=use_mlp,
            perplexity=perplexity,
            in_channels=inputs["data"].view_dims,
        ).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        best_mse = 999
        best_acc = -1
        best_outputs = None

        for epoch in range(epochs):
            self.model.train()

            for *views, mask in tqdm(inputs['train_dataloader'], desc=f'Train {epoch + 1:04d}'):
                x = self.preprocess_minibatch(
                    X_view=views,
                    M=mask,
                    perplexity=perplexity,
                    device=device,
                    data=inputs['data'],
                )

                x = self.model(x)
                loss = self.loss(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (1 + epoch) % eval_epochs == 0:
                metrics, outputs = self.evaluate(
                    dataloader=inputs['test_dataloader'],
                    device=device,
                    inputs=inputs,
                    ppl=perplexity,
                )
                mm.update(**metrics)
                logging.info(
                    f"epoch {epoch:04} {mm.report(current=True)}"
                )
                if metrics['MSE'] < best_mse or metrics["ACC"] > best_acc:
                    best_mse = metrics['MSE']
                    best_acc = metrics['ACC']
                    best_outputs = outputs

                if save_history:
                    self.history.append(metrics)
                if save_model:
                    torch.save(self.model.state_dict(), self.savedir.joinpath("best_model.pth"))

        best_outputs["history"] = self.history
        best_outputs['mm'] = mm
        best_outputs['X_gt'] = inputs['X_gt']
        return self.postprocess(best_outputs, save_vars)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, device, ppl: int, inputs: dict):
        GPU_EVAL = True
        self.model.eval()
        outputs = []
        loss = 0
        for *views, mask in tqdm(dataloader, desc=f'Eval(GPU={int(GPU_EVAL)})'):
            x = self.preprocess_minibatch(
                M=mask,
                X_view=views,
                device=device,
                perplexity=ppl,
                data=self.data,
            )
            x = self.model(x)
            loss += self.loss(x).item()
            y = dict(X_hat=x['X_hat'], H_common=x['H_common'])
            # 为了从GPU上拿下来，减少显存的占用量。
            outputs.append(convert_numpy(y) if not GPU_EVAL else y)

        concatenate = torch.cat if GPU_EVAL else np.concatenate
        H_common = concatenate([item['H_common'] for item in outputs], 0)
        X_hat = [concatenate([
            item['X_hat'][v] for item in outputs
        ], 0) for v in range(self.data.viewNum)]
        if GPU_EVAL:
            X_gt = convert_tensor(inputs['X_gt'], dev=device)
            M = convert_tensor(inputs['M'], dev=device, dtype=torch.bool)
        else:
            X_gt = inputs['X_gt']
            M = inputs['M']

        loss /= len(dataloader)
        mse = mse_missing_part(X_hat=X_hat, X=X_gt, M=~M)
        metrics = KMeans_Evaluate(H_common, self.data)
        metrics.update(MSE=mse, loss=loss)
        outputs = dict(H_common=H_common, X_hat=X_hat)
        outputs = convert_numpy(outputs)
        return metrics, outputs

    def postprocess(self, outputs: dict, save_vars: bool):
        mm: MaxMetrics = outputs["mm"]
        metrics = mm.report(current=False)
        if save_vars:
            for name in "history H_common X_hat X_gt".split():
                save_var(self.savedir, convert_numpy(outputs[name]), name)

        return metrics

    def preprocess(
        self,
        datapath: P,
        views,
        eta: float,
        shuffle: bool,
        batch_size: int,
    ):
        paired_rate: float = 1 - eta

        data = PartialMultiviewDataset(
            datapath=datapath,
            view_ids=views,
            paired_rate=paired_rate,
            partial_kind='partial',
            normalize='minmax',
        )
        logging.info(
            "Loaded dataset {}, #views {} paired_rate {}".format(
                data.name,
                data.viewNum,
                paired_rate,
            )
        )

        ptdata = PyTorchPartialMultiviewDataset.from_partial_multiview_dataset(data)

        train_dataloader = DataLoader(
            dataset=ptdata,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        test_dataloader = DataLoader(
            dataset=ptdata,
            shuffle=False,
            batch_size=batch_size,
        )
        res = dict(
            data=data,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            viewNum=data.viewNum,
            X_gt=data.X_gnd,
            M=data.mask,
            mm=MaxMetrics(MSE=False, ACC=True, NMI=True, PUR=True, F1=True),
        )

        return res

    def preprocess_minibatch(
        self,
        data,
        X_view: List[Tensor],
        M: Tensor,
        perplexity: int,
        device: str,
    ):
        X_view = [X_view[v][M[:, v]] for v in range(data.viewNum)]
        X_view = [X.to(device) for X in X_view]
        M = M.to(device)
        S_view = [
            calculate_optimized_p_cond(x, math.log2(perplexity), dev=device)
            for x in X_view
        ]

        P_view = [make_joint(s) for s in S_view]

        res = dict(
            data=data,
            viewNum=data.viewNum,
            M=M,
            S_view=S_view,
            P_view=P_view,
            X_view=X_view,
        )

        return res
