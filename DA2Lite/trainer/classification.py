import os
from typing import List, Tuple
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader

from DA2Lite.trainer.common import TrainerBase
from DA2Lite.core.log import get_logger

logger = get_logger(__name__)


class Classification(TrainerBase):
    def __init__(
        self,
        cfg_util,
        train_cfg,
        prefix,
        model,
        train_dataset,
        test_dataset,
        device,
        **kwargs,
    ):

        super().__init__(cfg_util, prefix, train_dataset, test_dataset, device)

        self.origin_summary = None
        if "origin_summary" in kwargs:
            self.origin_summary = kwargs["origin_summary"]

        self.model = model
        self.train_cfg = train_cfg
        self.is_train = train_cfg.IS_USE
        cfg_util.train_config = train_cfg  # change to train config of compression
        self.epochs = train_cfg.EPOCHS

        # Distributed training settings
        dist_cfg = cfg_util.cfg.DISTRIBUTED
        dist_cfg.defrost()

        if dist_cfg.URL == "env://" and dist_cfg.WORLD_SIZE == -1:
            dist_cfg.WORLD_SIZE = int(os.environ["WORLD_SIZE"])

        dist_cfg.IS_USE = dist_cfg.WORLD_SIZE > 1 or dist_cfg.IS_USE

        ngpus_per_node = torch.cuda.device_count()
        if dist_cfg.IS_USE:
            dist_cfg.WORLD_SIZE = ngpus_per_node * dist_cfg.WORLD_SIZE

            mp.spawn(
                self._run,
                nprocs=ngpus_per_node,
                args=(ngpus_per_node, self, cfg_util),
            )
        else:
            self._run(cfg_util.get_device(), ngpus_per_node, self, cfg_util)

    @staticmethod
    def _run(device, ngpus_per_node, trainer, cfg_util):
        dist_cfg = cfg_util.cfg.DISTRIBUTED

        trainer.device = device
        model = cfg_util.load_model()

        if trainer.device is not None:
            logger.info(f"Use GPU: {trainer.device} for trining.")

        if cfg_util.DISTRIBUTED.IS_USE:
            if dist_cfg.URL == "env://" and dist_cfg.RANK == -1:
                dist_cfg.RANK = int(os.environ["RANK"])
            dist_cfg.RANK = dist_cfg.RANK * ngpus_per_node + device

            dist.init_process_group(
                backend=dist_cfg.BACKEND,
                init_method=dist_cfg.URL,
                world_size=dist_cfg.WORLD_SIZE,
                rank=dist_cfg.RANK,
            )

        if not torch.cuda.is_available():
            print("using CPU, this will be slow")
        elif cfg_util.DISTRIBUTED.IS_USE:
            if device is not None:
                torch.cuda.set_device(device)
                model.cuda(device)
                cfg_util.TRAIN.BATCH_SIZE = int(
                    cfg_util.TRAIN.BATCH_SIZE / ngpus_per_node
                )
                cfg_util.TRAIN.WORKERS = int(
                    (cfg_util.TRAIN.WORKERS + ngpus_per_node - 1) / ngpus_per_node
                )
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[device]
                )
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif device is not None:
            torch.cuda.set_device(device)
            model = model.cuda(device)
        else:
            model = torch.nn.DataParallel(model).cuda()

        trainer.model = model

        trainer.optimizer = cfg_util.get_optimizer(trainer.model)
        trainer.loss = cfg_util.get_loss()

        trainer.scheduler = None
        if trainer.train_cfg.SCHEDULER:
            trainer.scheduler = cfg_util.get_scheduler(trainer.optimizer)

        if cfg_util.DISTRIBUTED.IS_USE:
            sampler = torch.utils.data.distributed.DistributedSampler(
                trainer.train_dataset
            )
        else:
            sampler = None

        train_loader = DataLoader(
            trainer.train_dataset,
            batch_size=cfg_util.TRAIN.BATCH_SIZE,
            shuffle=(sampler is None),
            num_workers=cfg_util.TRAIN.WORKERS,
            pin_memory=True,
            sampler=sampler,
        )
        test_loader = DataLoader(
            trainer.test_dataset,
            batch_size=cfg_util.TRAIN.BATCH_SIZE,
            num_workers=cfg_util.TRAIN.WORKERS,
            pin_memory=True,
        )

        trainer.train_loader = train_loader
        trainer.test_loader = test_loader

        trainer.build(ngpus_per_node)

        dist.destroy_process_group()

    def train(self, epoch):

        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        total_correct = 0
        batches = 0
        loop = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader) - 1, leave=False
        )
        for i, (images, labels) in loop:
            batches += len(labels)

            images, labels = Variable(images).to(self.device), Variable(labels).to(
                self.device
            )
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.loss(outputs, labels)

            pred = outputs.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

            loss.backward()
            self.optimizer.step()

            acc = float(total_correct) / batches
            loop.set_description(f"Train - Epoch [{epoch}/{self.epochs}]")
            loop.set_postfix(Accuracy=acc, Loss=loss.item())

            if i == len(self.train_loader) - 1:
                logger.debug(
                    f"Train - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {loss.item()}"
                )

    def test(self, epoch, print_log=True):

        self.model.eval()

        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                avg_loss += self.loss(outputs, labels)
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= i + 1
        acc = float(total_correct) / len(self.test_loader.dataset)

        if epoch != -1 and print_log == True:
            logger.debug(
                f"Test  - Epoch [{epoch}/{self.epochs}] Accuracy: {acc}, Loss: {avg_loss.data.item()}"
            )
        # else:
        #    logger.info(f'Test Accuracy: {acc}, Loss {avg_loss.data.item()}')
        return acc, avg_loss.data.item()

    def evaluate(self, print_log=True):
        return self.test(-1, print_log)

    def build(self, ngpus_per_node):
        logger.info(f"loading {self.prefix}_{self.model_name}...")

        if self.is_train:
            self._print_train_cfg()
            for epoch in range(1, self.epochs + 1):
                if self.cfg.DISTRIBUTED.IS_USE:
                    self.train_loader.sampler.set_epoch(epoch)
                self.train(epoch)
                test_acc, test_loss = self.test(epoch)

                if self.scheduler != None:
                    self.scheduler.step()
        else:
            test_acc, test_loss = self.evaluate()
        logger.info(f"The trained model is saved in {self.save_path}\n")
        if not self.cfg.DISTRIBUTED.IS_USE or (
            self.cfg.DISTRIBUTED.IS_USE
            and self.cfg.DISTRIBUTED.RANK % ngpus_per_node == 0
        ):
            torch.save(self.model, self.save_path)

        summary_dict = self.model_summary(
            test_acc, test_loss, self.model, self.origin_summary
        )

        self.cleanup_ddp()

        return self.model, summary_dict

    def _print_train_cfg(self):
        split_train_cfg = str(self.train_cfg).split("\n")

        num_dummy = 60
        train_txt = " Train configuration ".center(num_dummy, " ")
        border_txt = "-" * num_dummy

        logger.info(f"+{border_txt}+")
        logger.info(f"|{train_txt}|")
        logger.info(f"+{border_txt}+")
        logger.info(f'|{" ".ljust(num_dummy)}|')
        for i_tr_cfg in split_train_cfg:
            if "IS_USE" in i_tr_cfg:
                continue
            logger.info(f"| {i_tr_cfg.ljust(num_dummy-1)}|")
        logger.info(f'|{" ".ljust(num_dummy)}|')
        logger.info(f"+{border_txt}+\n")
