"""Tweaked version of corresponding AllenNLP file"""
import datetime
import logging
import math
import os
import time
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any

import torch
import torch.optim.lr_scheduler
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb, lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase

logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        iterator: DataIterator,
        train_dataset: Iterable[Instance],
        validation_dataset: Optional[Iterable[Instance]] = None,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        accumulated_batch_count: int = 1,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: Union[int, List] = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        cold_step_count: int = 0,
        cold_lr: float = 1e-3,
        cuda_verbose_step=None,
    ) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a ``DataIterator``, and uses the supplied ``Optimizer`` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        checkpointer : ``Checkpointer``, optional (default=None)
            An instance of class Checkpointer to use instead of the default. If a checkpointer is specified,
            the arguments num_serialized_models_to_keep and keep_serialized_model_every_num_seconds should
            not be specified. The caller is responsible for initializing the checkpointer so that it is
            consistent with serialization_dir.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``Union[int, List[int]]``, optional (default = -1)
            An integer or list of integers specifying the CUDA device(s) to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``LearningRateScheduler``, optional (default = None)
            If specified, the learning rate will be decayed with respect to
            this schedule at the end of each epoch (or batch, if the scheduler implements
            the ``step_batch`` method). If you use :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the ``validation_metric`` provided to determine if learning has plateaued.
            To support updating the learning rate on every batch, this can optionally implement
            ``step_batch(batch_num_total)`` which updates the learning rate given the batch number.
        momentum_scheduler : ``MomentumScheduler``, optional (default = None)
            If specified, the momentum will be updated at the end of each batch or epoch
            according to the schedule.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : ``int``, optional, (default = ``None``)
            If defined, how often to log the average batch size.
        moving_average: ``MovingAverage``, optional, (default = None)
            If provided, we will maintain moving averages for all parameters. During training, we
            employ a shadow variable for each parameter, which maintains the moving average. During
            evaluation, we backup the original parameters and assign the moving averages to corresponding
            parameters. Be careful that when saving the checkpoint, we will save the moving averages of
            parameters. This is necessary because we want the saved model to perform as well as the validated
            model if we load it later. But this may cause problems if you restart the training from checkpoint.
        """
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.iterator = iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataset
        self._validation_data = validation_dataset
        self.accumulated_batch_count = accumulated_batch_count
        self.cold_step_count = cold_step_count
        self.cold_lr = cold_lr
        self.cuda_verbose_step = cuda_verbose_step

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(patience)
            )

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        if checkpointer is not None:
            # We can't easily check if these parameters were passed in, so check against their default values.
            # We don't check against serialization_dir since it is also used by the parent class.
            if num_serialized_models_to_keep != 20 \
                    or keep_serialized_model_every_num_seconds is not None:
                raise ConfigurationError(
                    "When passing a custom Checkpointer, you may not also pass in separate checkpointer "
                    "args 'num_serialized_models_to_keep' or 'keep_serialized_model_every_num_seconds'."
                )
            self._checkpointer = checkpointer
        else:
            self._checkpointer = Checkpointer(
                serialization_dir,
                keep_serialized_model_every_num_seconds,
                num_serialized_models_to_keep,
            )

        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: self._batch_num_total,
            serialization_dir=serialization_dir,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        self._log_batch_size_period = log_batch_size_period

        self._last_log = 0.0  # time of last logging

        # Enable activation logging.
        if histogram_interval is not None:
            self._tensorboard.enable_activation_logging(self.model)

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, self.model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])

            # 这里面我们进行强制自己定义学习=true,
            # for i in  list(self.model.parameters()):
            #     i.requires_grad=True

            output_dict = self.model(**batch) # 里面是训练过程.

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty() # 保持泛化性.这里面没设置,所以是惩罚系数=0
        except KeyError:
            if for_training:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output of model.forward(inputs)."
                )
            loss = None

        return loss

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        # logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        # logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)  # 如果没有gpu ,也返回1.

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data, num_epochs=1, shuffle=self.shuffle) # 这一步生成bert_off    这个东西生成的含义是他里面的数字代表我们每一个编码的起始位置.

        '''
        [$START, 我, 今, 天, 去, 了, 金, 子, 塔, 去, 玩, apple, 欧, 耶]
  tensor([[ 4,  8,  9, 11, 12, 14, 16, 17, 19, 20, 22, 24, 27, 29],
  
  表示 开始start对应编码后数组索引是4到7,我对应8到8,今对应9到10
        '''

        #          train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data) / num_gpus)
        residue = num_training_batches % self.accumulated_batch_count
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        # train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches) # 打印一个进度条而已.
        train_generator_tqdm = train_generator # 打印一个进度条而已.
        cumulative_batch_size = 0
        self.optimizer.zero_grad()
        print('训练一个epoch')
        for batch_group in train_generator_tqdm:
            #




            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            iter_len = self.accumulated_batch_count \
                if batches_this_epoch <= (num_training_batches - residue) else residue
            if 1:
                iter_len=1


            try:
                loss = self.batch_loss(batch_group, for_training=True) / iter_len # 输入的数据里面去除了全部都是keep的情况
            except RuntimeError as e:
                print(e)
                for x in batch_group:
                    all_words = [len(y['words']) for y in x['metadata']]
                    print(f"Total sents: {len(all_words)}. "
                          f"Min {min(all_words)}. Max {max(all_words)}")
                    for elem in ['labels', 'd_tags']:
                        tt = x[elem]
                        print(
                            f"{elem} shape {list(tt.shape)} and min {tt.min().item()} and {tt.max().item()}")
                    for elem in ["bert", "mask", "bert-offsets"]:
                        tt = x['tokens'][elem]
                        print(
                            f"{elem} shape {list(tt.shape)} and min {tt.min().item()} and {tt.max().item()}")
                raise e

            # if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
            #     print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
            #     print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()

            # if self.cuda_verbose_step is not None and batch_num_total % self.cuda_verbose_step == 0:
            #     print(f'After backprop - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
            #     print(f'After backprop - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            train_loss += loss.item() * iter_len

            del batch_group, loss
            torch.cuda.empty_cache() # 节省内存,显存
            batch_grad_norm = self.rescale_gradients()
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)
            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }
                if batches_this_epoch % self.accumulated_batch_count == 0 or \
                        batches_this_epoch == num_training_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1))
                    param_norm = torch.norm(param.view(-1)).cpu()
                    self._tensorboard.add_train_scalar(
                        "gradient_update/" + name, update_norm / (param_norm + 1e-7)
                    )
            else:
                if batches_this_epoch % self.accumulated_batch_count == 0 or \
                        batches_this_epoch == num_training_batches:
                    self.optimizer.step()   #多个batch才进行bp算法.
                    self.optimizer.zero_grad()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)
        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)

        raw_val_generator = val_iterator(self._validation_data, num_epochs=1, shuffle=True)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(
            val_iterator.get_num_batches(self._validation_data) / num_gpus
        )
        val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:

            loss = self.batch_loss(batch_group, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, batches_this_epoch

    def train(self,oldmodel) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """#---------------通过oldmodel来加载官方模型,从而进行finetune
        epoch_counter = 0
        try:
            # epoch_counter = self._restore_checkpoint()
            epoch_counter=0 # 直接进行finetune. 加速训练.
            # 下面看这个地方怎么修改,保证之前的参数都赋值上去.?????????????????????????????????????之前没遇到过这个问题,需要自己克服看看.
            print(oldmodel,'旧的模型路径是这个')  # 这个model里面已经涵盖了xlnet网络的参数和最后2层linear的参数.
            tmp=torch.load(oldmodel, map_location=torch.device('cpu'))
            out_shape = self.model.tag_labels_projection_layer._module.out_features # 把下面的东西扩充到out_shape
            now_shape=tmp['tag_labels_projection_layer._module.weight'].shape[0]
            fix_shape=out_shape-now_shape
            # 通过concat
            tmp['tag_labels_projection_layer._module.weight']=torch.cat((tmp['tag_labels_projection_layer._module.weight'],torch.zeros(fix_shape,768)),  0)
            tmp['tag_labels_projection_layer._module.bias']=torch.cat((tmp['tag_labels_projection_layer._module.bias'],torch.zeros(fix_shape)),  0)
            # 需要补充到的数据大小:



            # tmp只是一个字典而已,随便玩.

            self.model.load_state_dict(tmp)# 这次的收敛速度飞快!!!!!!!!!!!!!!!!!!!# 初步的打算是,补充shape到我们需要的,大小.

        except :
            print('读取旧模型'+oldmodel+' 失败,下面开始从头训练.')

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()
        base_lr_save = self.optimizer.param_groups[0]['lr']
        if self.cold_step_count > 0: # 冷启动几个step, 这些step不用更新权重.对于embed网路都freeeze上.只更新后面自己简历的分类器网络.
            base_lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cold_lr
            self.model.text_field_embedder._token_embedders['bert'].set_weights(freeze=True)
        base_lr = base_lr_save
        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value
        print('总共需要学习的轮次',self._num_epochs)
        for epoch in range(epoch_counter, self._num_epochs):# 把之前学完的epoch直接跳过.
            if epoch == self.cold_step_count and epoch != 0: # 冷启动完毕,开始恢复学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = base_lr
                self.model.text_field_embedder._token_embedders['bert'].set_weights(freeze=False)#并且把embed网络解除冻结

            epoch_start_time = time.time()# 下行是训练代码--------!!!!!!!!!!!!!!!!
            print('当前epoch',epoch,'当前学习率',param_group['lr'],)
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            # clear cache before validation
            torch.cuda.empty_cache()


            # 在验证集上评测效果,防止过拟合.

            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss() # 下面一行计算准确率
                    val_metrics = training_util.get_metrics(
                        self.model, val_loss, num_batches, reset=True
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)
                    if 0:
                     if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        print('early strop!!!!!!!')
                        break

            # self._tensorboard.log_metrics(
            #     train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
            # )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            # if self.cold_step_count <= epoch:
            self.scheduler.step(metrics['validation_loss'])

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir:
                dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric, epoch)
#保存model, 只需要给定这个文件夹,那么算法自动会读取里面最新的模型.来进行finetune.很方便.


            if self._num_epochs==epoch+1: # 只存最后一个.
              print('进行最后一个epoch了!!!!!!!!!')
              self.model.eval()
              class a():
                  pass
              args=a()
              args.model_dir='tmpModel2zh'
              # self._save_checkpoint(epoch) # 每一个epoch 都存,最后的磁盘占用很大.
              out_model = os.path.join(args.model_dir, 'model.th')
              with open(out_model, 'wb') as f:
                  torch.save(self.model.state_dict(), f)
              print("Model is dumped!!!!!!!!!!!", "训练全部结束,model存在了", args.model_dir + ' / model.th')

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            # if epoch < self._num_epochs - 1:
            #     training_elapsed_time = time.time() - training_start_time
            #     estimated_time_remaining = training_elapsed_time * (
            #         (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
            #     )
            #     formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
            #     logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        # self._tensorboard.close()

        # Load the best model state before returning # 根据路径目录找到里面存的最好的模型.
        # best_model_state = self._checkpointer.best_model_state()
        # if best_model_state:
        #     self.model.load_state_dict(best_model_state)

        return metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total,
        }

        # If we have a learning rate or momentum scheduler, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()


        if self._metric_tracker.is_best_so_far():# 把save改成只存一份最好的,不好的直接pass.节省磁盘空间.
            print('得到最优模型,正在保存')
            self._checkpointer.save_checkpoint(
                model_state=self.model.state_dict(),
                epoch=epoch,
                training_states=training_states,
                is_best_so_far=self._metric_tracker.is_best_so_far(),
            )


        # Restore the original values for parameters so that training will not be affected.
        if self._moving_average is not None:
            self._moving_average.restore()

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """#就是去目录里面找权重文件,然后拿过来继续训练.
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None \
                and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metric_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split(".")[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get("batch_num_total")
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(  # type: ignore
        cls,
        model: Model,
        serialization_dir: str,
        iterator: DataIterator,
        train_data: Iterable[Instance],
        validation_data: Optional[Iterable[Instance]],
        params: Params,
        validation_iterator: DataIterator = None,
    ) -> "Trainer":

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(
                params.pop("moving_average"), parameters=parameters
            )
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if "checkpointer" in params:
            if "keep_serialized_model_every_num_seconds" in params \
                    or "num_serialized_models_to_keep" in params:
                raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods."
                )
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None
            )
            checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
            )
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(
            model,
            optimizer,
            iterator,
            train_data,
            validation_data,
            patience=patience,
            validation_metric=validation_metric,
            validation_iterator=validation_iterator,
            shuffle=shuffle,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=lr_scheduler,
            momentum_scheduler=momentum_scheduler,
            checkpointer=checkpointer,
            model_save_interval=model_save_interval,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
            log_batch_size_period=log_batch_size_period,
            moving_average=moving_average,
        )
