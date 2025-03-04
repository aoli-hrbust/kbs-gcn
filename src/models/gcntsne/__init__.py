from src.models.gcntsne.train import CompletionTrainer, P, EasyDict, train_begin, train_end
from src.models.gcntsne.train_minibatch import MinibatchCompletionTrainer
from src.utils.torch_utils import get_device
from src.utils.io_utils import *


def train_main(
    datapath=P("./data/ORL-40.mat"),
    eta=0.5,
    views=None,
    perplexity: int = 10,
    lamda: float = 0.1,
    use_mlp: bool = False,
    epochs: int = 200,
    eval_epochs: int = 10,
    hidden_dims: int = 128,
    lr: float = 0.001,
    batch_size: int = 128,
    shuffle: bool = False,
    use_minibatch: bool = False,
    before: bool = False,
    device=get_device(),
    savedir: P = P("output/debug"),
    save_vars: bool = False,
    save_history: bool = False,
    save_model: bool = False,
    inference_only: bool = False,
):
    args = EasyDict(
        datapath=datapath,
        eta=eta,
        views=views,
        perplexity=perplexity,
        lamda=lamda,
        use_mlp=use_mlp,
        epochs=epochs,
        eval_epochs=eval_epochs,
        lr=lr,
        use_minibatch=use_minibatch,
        batch_size=batch_size,
        hidden_dims=hidden_dims,
        shuffle=shuffle,
        device=device,
        savedir=savedir,
        save_vars=save_vars,
        save_history=save_history,
        save_model=save_model,
        method="GTSNE-MvAE",
    )

    if use_minibatch:
        train_begin(savedir, args, "Begin GTSNE-MvAE (minibatch) training...")
        Trainer = MinibatchCompletionTrainer
    else:
        train_begin(savedir, args, "Begin GTSNE-MvAE (wholly-batch) training...")
        Trainer = CompletionTrainer

    trainer = Trainer(
        lamda=lamda,
        before=before,
        savedir=savedir,
    )
    if inference_only:
        metrics = trainer.inference(
            datapath=datapath,
            views=views,
            eta=eta,
            perplexity=perplexity,
            use_mlp=use_mlp,
            epochs=epochs,
            eval_epochs=eval_epochs,
            lr=lr,
            hidden_dims=hidden_dims,
            batch_size=batch_size,
            device=device,
        )
    else:
        metrics = trainer.train(
            datapath=datapath,
            views=views,
            eta=eta,
            perplexity=perplexity,
            use_mlp=use_mlp,
            epochs=epochs,
            eval_epochs=eval_epochs,
            lr=lr,
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
            save_vars=save_vars,
            hidden_dims=hidden_dims,
            save_history=save_history,
            save_model=save_model,
        )
    if save_model:
        metrics = trainer.inference(
            datapath=datapath,
            views=views,
            eta=eta,
            perplexity=perplexity,
            use_mlp=use_mlp,
            lr=lr,
            hidden_dims=hidden_dims,
            batch_size=batch_size,
            device=device,
        )

    train_end(savedir, metrics=metrics)
