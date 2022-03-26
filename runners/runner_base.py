from ipaddress import collapse_addresses
import os

import torch
import random, time, json, datetime
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import utils.blip_utils as utils

import yaml
from pathlib import Path
from common.registry import registry


class Runner():
    def __init__(self, cfg, task, model, datasets):
        self.config = cfg

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._dataloaders = None

        self.setup_seeds()
        self.setup_output_dir()


    def setup_seeds(self):
        seed = self.config.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.device)
        
        return self._device

    @property
    def use_distributed(self):
        return self.config.distributed

    @property
    def model(self):
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = torch.nn.parallel.DistributedDataParallel(
                                            self._model, 
                                            device_ids=[self.config.gpu]
                                        )
            else:
                self._wrapped_model = self._model
                    
        return self._wrapped_model
         
    @property
    def model_without_ddp(self):
        if self.use_distributed:
            return self.model.module
        else:
            return self.model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                params=self.model.parameters(), 
                lr=float(self.config.init_lr),
                weight_decay=float(self.config.weight_decay)
            )
        
        return self._optimizer

    @property
    def dataloaders(self):
        if self._dataloaders is None:

            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_train = [split in self.train_splits for split in split_names]

            if self.use_distributed:
                samplers = utils.create_sampler(
                    datasets=datasets,
                    shuffles=is_train,
                    num_tasks=utils.get_world_size(), 
                    global_rank=utils.get_rank()
                )
            else:
                samplers = [None] * len(self.datasets)

            dataloaders = utils.create_loader(
                datasets=datasets,
                samplers=samplers,
                batch_size=[self.config.batch_size] * len(datasets),
                num_workers=[self.config.num_workers] * len(datasets),
                is_trains=is_train,
                collate_fns=[None] * len(datasets)
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}
        
        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return self.config.max_epoch

    @property
    def valid_splits(self):
        return self.config.valid_splits

    @property
    def train_splits(self):
        return self.config.train_splits

    @property
    def evaluate_only(self):
        return self.config.evaluate

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.output_dir 
        result_dir = output_dir / 'result'

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir
        
        
    def train(self):
        best = 0
        best_epoch = 0

        # print("Start training")
        start_time = time.time()
        for epoch in range(0, self.max_epoch):
            # if not self.args.evaluate:
            #     if self.args.distributed:
            #         self.train_loader.sampler.set_epoch(epoch)

            #     utils.cosine_lr_schedule(self.optimizer, epoch, int(self.config['max_epoch']), float(self.config['init_lr']), float(self.config['min_lr']))

            #     train_stats = self.train(epoch)

            for split_name in self.valid_splits:
                val_result = self.validate(split_name=split_name)

                self.task.after_validation(val_result=val_result, split_name=split_name, epoch=epoch)

            if self.evaluate_only:
                break
            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    # def train(self, epoch):
    #     # train
    #     self.model.train()  
        
    #     metric_logger = utils.MetricLogger(delimiter="  ")
    #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #     metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    #     header = 'Train Caption Epoch: [{}]'.format(epoch)
    #     print_freq = 50

    #     for i, (image, caption, _) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
    #         image = image.to(self.device)
            
    #         loss = self.model(image, caption)      
            
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()    
            
    #         metric_logger.update(loss=loss.item())
    #         metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

    #     # gather the stats from all processes
    #     metric_logger.synchronize_between_processes()
    #     print("Averaged stats:", metric_logger.global_avg())     
    #     return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

    @torch.no_grad()
    def validate(self, split_name):
        # TODO In validation, you need to compute loss as well as metrics
        model = self.model_without_ddp
        model.eval()

        data_loader = self.dataloaders.get(split_name, None)

        assert data_loader, "data_loader for split {} is None.".format(split_name)
        
        # TODO doesn't look like a good place to define logger
        # Possibly called multiple times on different splits.
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Caption generation:'
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header): 
            samples = self._prepare_sample(samples)
            
            eval_output = self.task.valid_step(model=model, samples=samples)
            results.extend(eval_output)
    
        return results

    def _prepare_sample(self, samples):
        if self.cuda_enabled:
            samples = utils.move_to_cuda(samples)
        
        # TODO fp16 support

        return samples