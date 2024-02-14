# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import json

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR
import pts
from gluonts.evaluation import make_evaluation_predictions, Evaluator, MultivariateEvaluator


class BestValTrainer(pts.Trainer):
    def __init__(
        self,
        savepath,
        epochs=None,
        n_batches_val=None,
        load=0,
        cont=False,
        weight_decay=0,
        eps=1e-8,
        last_model_filename="last.pt",
        best_model_filename="best.pt",
        metrics_filename="metrics.csv",
        onecycle=False,
        val_freq=10,
        early_stopping=False,
        patience=10,
        multi_dim=True,
        use_best=True,
        **kwargs,
    ):
        self.onecycle = onecycle
        print(f"epochs = {epochs}")
        print(f"onecycle = {onecycle}")
        if onecycle:
            assert epochs is not None
        elif epochs is None:
            epochs = 1000000
            early_stopping = True
        print(f"epochs = {epochs}, early_stopping = {early_stopping}")
        super().__init__(weight_decay=weight_decay, epochs=epochs, **kwargs)
        self.savepath = savepath
        self.last_model_filename = last_model_filename
        self.best_model_filename = best_model_filename
        self.metrics_filename = metrics_filename
        self.n_batches_val = n_batches_val
        self.load = load
        self.cont = cont
        self.early_stopping = early_stopping
        self.patience = patience
        self.lr_patience = patience
        self.use_best = use_best
        self.eps = eps
        self.val_freq = val_freq
        quantiles = (np.arange(20)/20.0)[1:]
        if multi_dim:
            self.evaluator = MultivariateEvaluator(
                quantiles=quantiles,
                target_agg_funcs={'sum': np.sum}
            )
        else:
            self.evaluator = Evaluator(quantiles=quantiles)
    
    def save_last(self, net):
        fpath = self.savepath / self.last_model_filename
        print(f"Save last model to {fpath}.")
        torch.save(net.state_dict(), fpath)

    def save_best(self, net):
        fpath = self.savepath / self.best_model_filename
        print(f"Save best model to {fpath}.")
        torch.save(net.state_dict(), fpath)

    def load_last(self, net):
        fpath = self.savepath / self.last_model_filename
        print(f"Load last model from {fpath}.")
        net.load_state_dict(torch.load(fpath)) 
        print("Continue training")

    def load_best(self, net):
        fpath = self.savepath / self.best_model_filename
        print(f"Load best model from {fpath}.")
        net.load_state_dict(torch.load(fpath)) 

    def log_metrics(self, epoch, **kwargs):
        keys = list(kwargs.keys())
        metrics = [str(epoch)] + [str(kwargs[k]) for k in keys]
        if epoch == 1:
            columns = ["epoch"] + keys
            with open(self.savepath / self.metrics_filename, "w") as f:
                print(",".join(columns), file=f)
        with open(self.savepath / self.metrics_filename, "a") as f:
            print(",".join(metrics), file=f)

    def try_load(self, net, load):
        if load > 1:
            self.load_last(net)
            return not self.cont
        elif load == 1:
            self.load_best(net)
            return not self.cont
        else:
            return False
    
    def _train_begin(self, net):
        self.optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, eps=self.eps,
        )
        if self.onecycle:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.maximum_learning_rate,
                steps_per_epoch=self.num_batches_per_epoch,
                epochs=self.epochs,
                verbose=False,
            )
        else:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.lr_patience, verbose=True)

    def _train_one_epoch(self, epoch_no, net, train_iter):
        total = self.num_batches_per_epoch
        cum_loss = 0.0
        net.train()
        with tqdm(train_iter, total=total) as it:
            for batch_no, data_entry in enumerate(it, start=1):
                if batch_no > self.num_batches_per_epoch:
                    break
                self.optimizer.zero_grad()
                inputs = [v.to(self.device) for v in data_entry.values()]
                output = net(*inputs)
                if isinstance(output, (list, tuple)):
                    loss = output[0]
                else:
                    loss = output
                cum_loss += loss.item()
                avg_loss = cum_loss / batch_no
                it.set_postfix(
                    {
                        "epoch": f"{epoch_no}/{self.epochs}",
                        "avg_loss": avg_loss,
                    },
                    refresh=False,
                )
                loss.backward()
                self.optimizer.step()
                if self.onecycle:
                    self.scheduler.step()
            it.close()
        return avg_loss, {"lr": self.optimizer.param_groups[0]["lr"]}

    def _epoch_end(self, loss):
        if not self.onecycle:
            self.scheduler.step(loss)

    def __call__(
        self,
        net,
        train_iter,
        validation_data=None,
        predictor_creator=None,
        validation_iter=None,
    ):
        if self.try_load(net, self.load):
            return

        self._train_begin(net)
        best_loss = None
        best_epoch = None
        for epoch_no in range(1, self.epochs+1):
            loss_train, metrics_train = self._train_one_epoch(epoch_no, net, train_iter)
            net.eval()
            if validation_data is not None and epoch_no % self.val_freq == 0:
                loss_val = self.eval(net, validation_data, predictor_creator, epoch_no)
            elif validation_iter is not None and epoch_no % self.val_freq == 0:
                loss_val = self.eval_as_train(net, validation_iter)
            else:
                loss_val = None
            self._epoch_end(loss_train)
            print(f"End of epoch {epoch_no}")
            self.save_last(net)
            if loss_val is not None:
                if best_loss is None or best_loss > loss_val:
                    patience = self.patience
                    print(f"Better loss: {loss_val}. Save model to {self.savepath}.")
                    self.save_best(net)
                    best_loss = loss_val
                    best_epoch = epoch_no
                else:
                    patience -= 1
                    print(f"Best loss: {best_loss} from epoch {best_epoch}, current loss: {loss_val}, patience: {patience}.")
                    if self.early_stopping and patience <= 0:
                        break
            self.log_metrics(
                epoch=epoch_no,
                train_loss=loss_train,
                val_loss=loss_val,
                **metrics_train,
            )
        print(f"Best epoch: {best_epoch}")
        if self.use_best and best_loss is not None:
            self.load_best(net)
        return best_epoch
    
    def eval(self, net, ds_val, predictor_creator, epoch):
        predictor = predictor_creator(net)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=ds_val,
            predictor=predictor,
            num_samples=100,
        )
        agg_metrics, item_metrics = self.evaluator(ts_it, forecast_it, num_series=len(ds_val))
        outpath = self.savepath / "epoch" / str(epoch)
        outpath.mkdir(parents=True, exist_ok=True)
        with open(outpath / "agg_metrics.json", "w") as f:
            json.dump(agg_metrics, f, indent=4)
        item_metrics.to_csv(outpath / "item_metrics.csv")
        return agg_metrics["mean_wQuantileLoss"]

    def eval_as_train(self, net, validation_iter):
        cum_loss_val = 0.0
        with tqdm(validation_iter, total=self.n_batches_val) as it:
            for batch_no, data_entry in enumerate(it):
                inputs = [v.to(self.device) for v in data_entry.values()]
                with torch.no_grad():
                    output = net(*inputs)
                if isinstance(output, (list, tuple)):
                    loss = output[0]
                else:
                    loss = output
                cum_loss_val += loss.item()
                avg_loss_val = cum_loss_val / (batch_no + 1)
                it.set_postfix(
                    {
                        "avg_loss_val": avg_loss_val,
                    },
                    refresh=False,
                )
                if self.n_batches_val is not None and batch_no >= self.n_batches_val:
                    break
        it.close()
        return avg_loss_val


class CoordBestValTrainer(BestValTrainer):
    def __init__(self, n_update=None, learning_rate_model=None, weight_decay_model=None, **kwargs):
        super().__init__(**kwargs)
        if n_update is None:
            n_update = self.num_batches_per_epoch
        self.n_update = n_update
        if learning_rate_model is None:
            learning_rate_model = self.learning_rate
        self.learning_rate_model = learning_rate_model
        if weight_decay_model is None:
            weight_decay_model = self.weight_decay
        self.weight_decay_model = weight_decay_model

    def _update_sample(self, net, train_iter):
        with tqdm(train_iter) as it:
            for batch_no, data_entry in enumerate(it, start=1):
                inputs = [v.to(self.device) for v in data_entry.values()]
                net.update_sample(*inputs)
    
    def _train_begin(self, net):
        posterior_params = list(net.get_posterior_params())
        if posterior_params:
            self.posterior_optimizer = Adam(posterior_params, lr=self.learning_rate, weight_decay=self.weight_decay, eps=self.eps)
            if self.onecycle:
                self.posterior_scheduler = OneCycleLR(
                    self.posterior_optimizer,
                    max_lr=self.maximum_learning_rate,
                    steps_per_epoch=self.n_update,
                    epochs=self.epochs,
                    verbose=True,
                )
            else:
                self.posterior_scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.posterior_optimizer, 'min', patience=self.lr_patience, verbose=True
                )
            self.update_posterior = True
        else:
            self.update_posterior = False
        model_params = list(net.get_model_params())
        if model_params:
            self.model_optimizer = Adam(model_params, lr=self.learning_rate_model, weight_decay=self.weight_decay_model, eps=self.eps)
            if self.onecycle:
                self.model_scheduler = OneCycleLR(
                    self.model_optimizer,
                    max_lr=self.maximum_learning_rate,
                    steps_per_epoch=self.num_batches_per_epoch,
                    epochs=self.epochs,
                    verbose=False,
                )
            else:
                self.model_scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.model_optimizer, 'min', patience=self.lr_patience, verbose=True
                )
            self.update_model = True
        else:
            self.update_model = False

    def _train_one_epoch(self, epoch_no, net, train_iter):
        net.train()
        posterior_train_iter, model_train_iter = train_iter
        lr = {}
        if self.update_model:
            net.update_model()
            self._update_sample(net, posterior_train_iter)
            avg_model_loss, cum_model_loss = self.train_model(
                net, model_train_iter, self.model_optimizer, epoch_no,
                total=self.num_batches_per_epoch, scheduler=self.model_scheduler)
            loss_train_all = (avg_model_loss, 0.0, 0.0)
            lr["model_lr"] = self.model_optimizer.param_groups[0]["lr"]
        if self.update_posterior:
            cum_post_loss = self.train_posterior(
                net, posterior_train_iter, self.posterior_optimizer, epoch_no,
                scheduler=self.posterior_scheduler)
            loss_train_all = cum_post_loss
            lr["posterior_lr"] = self.posterior_optimizer.param_groups[0]["lr"]
        return loss_train_all[0], {
            "loss_train": loss_train_all[0],
            "nll": loss_train_all[1],
            "kl": loss_train_all[2],
            **lr,
        }

    def _epoch_end(self, loss):
        if not self.onecycle:
            if self.update_posterior:
                self.posterior_scheduler.step(loss)
            if self.update_model:
                self.model_scheduler.step(loss)

    def train_posterior(self, net, train_iter, optimizer, epoch_no, scheduler=None):
        net.update_posterior()
        for batch_no, data_entry in enumerate(train_iter):
            inputs = [v.to(self.device) for v in data_entry.values()]
            ret_1 = net.training_forward_1(*inputs)
            full_dim_idx = net.get_full_dim_idx()
            n_dim_idx = len(full_dim_idx)
            for i_update in range(1, self.n_update+1):
                cum_loss_track = []
                full_dim_idx = full_dim_idx[torch.randperm(n_dim_idx)]
                with tqdm(torch.split(full_dim_idx, net.n_sample_dim)) as it:
                    for dim_idx in it:
                        net.reset()
                        net.set_dim_idx(dim_idx)
                        output = net.training_forward_2(*inputs, *ret_1)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output
                        if not cum_loss_track:
                            cum_loss_track = [0.0] * len(output)
                        for i, o in enumerate(output):
                            cum_loss_track[i] += o.item()
                        display = {
                            "epoch": f"{epoch_no}/{self.epochs}",
                            "update": f"{i_update}/{self.n_update}"
                        }
                        display.update({
                            f"cum_loss_{i}": l for i, l in enumerate(cum_loss_track)
                        })
                        it.set_postfix(display, refresh=False)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if self.onecycle:
                            scheduler.step()
                    it.close()
        return cum_loss_track

    def train_model(self, net, train_iter, optimizer, epoch_no, total=None, scheduler=None):
        cum_loss_track = []
        with tqdm(train_iter, total=total) as it:
            for batch_no, data_entry in enumerate(it):
                if total and batch_no == total:
                    break
                optimizer.zero_grad()

                inputs = [v.to(self.device) for v in data_entry.values()]
                output = net(*inputs)

                if isinstance(output, (list, tuple)):
                    loss = output[0]
                else:
                    loss = output

                if not cum_loss_track:
                    cum_loss_track = [0.0] * len(output)
                for i, o in enumerate(output):
                    cum_loss_track[i] += o.item()
                avg_loss_track = [cl / (batch_no + 1) for cl in cum_loss_track]
                display = {"epoch": f"{epoch_no}/{self.epochs}"}
                display.update({
                    f"avg_loss_{i}": l for i, l in enumerate(avg_loss_track)
                })
                display.update({
                    f"cum_loss_{i}": l for i, l in enumerate(cum_loss_track)
                })
                it.set_postfix(display, refresh=False)

                loss.backward()
                optimizer.step()
                if self.onecycle:
                    scheduler.step()

            it.close()
        return avg_loss_track[0], cum_loss_track[0]
