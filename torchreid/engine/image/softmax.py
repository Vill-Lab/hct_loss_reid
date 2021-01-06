from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime
import torch

from torchreid.engine import Engine
from torchreid.losses import CrossEntropyLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, optimizer, scheduler=None, use_gpu=True,
                 label_smooth=True):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.criterion_s = torch.nn.MSELoss()
        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses = AverageMeter()
        losses_x1 = AverageMeter()
        losses_x2 = AverageMeter()
        losses_x3 = AverageMeter()
        losses_x4 = AverageMeter()
        accs1 = AverageMeter()
        accs2 = AverageMeter()
        accs3 = AverageMeter()
        accs4 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.it = 0
        self.Threshold = 4

        self.model.train()
        if (epoch+1)<=fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            
            self.optimizer.zero_grad()
            output1, output2, output3, output4= self.model(imgs)
            loss_x1 = self._compute_loss(self.criterion, output1, pids)
            loss_x2 = self._compute_loss(self.criterion, output2, pids)
            loss_x3 = self._compute_loss(self.criterion, output3, pids)
            loss_x4 = self._compute_loss(self.criterion, output3, pids)

            if self.it % self.Threshold == 0:
                loss = loss_x2 + loss_x3 + loss_x4

            if self.it % self.Threshold == 1:
                loss = loss_x1 + loss_x3 + loss_x4

            if self.it % self.Threshold == 2:
                loss = loss_x1 + loss_x2 + loss_x4

            if self.it % self.Threshold == 3:
                loss = loss_x1 + loss_x2 + loss_x3
            self.it += 1

            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            losses.update(loss.item(), pids.size(0))

            losses_x1.update(loss_x1.item(), pids.size(0))
            losses_x2.update(loss_x2.item(), pids.size(0))
            losses_x3.update(loss_x3.item(), pids.size(0))
            losses_x4.update(loss_x4.item(), pids.size(0))
            accs1.update(metrics.accuracy(output1, pids)[0].item())
            accs2.update(metrics.accuracy(output2, pids)[0].item())
            accs3.update(metrics.accuracy(output3, pids)[0].item())
            accs4.update(metrics.accuracy(output4, pids)[0].item())


            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_x1 {loss_x1.val:.4f} ({loss_x1.avg:.4f})\t'
                      'Loss_x2 {loss_x2.val:.4f} ({loss_x2.avg:.4f})\t'
                      'Loss_x3 {loss_x3.val:.4f} ({loss_x3.avg:.4f})\t'
                      'Loss_x4 {loss_x4.val:.4f} ({loss_x4.avg:.4f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, num_batches,
                      loss=losses,
                      loss_x1=losses_x1,
                      loss_x2=losses_x2,
                      loss_x3=losses_x3,
                      loss_x4=losses_x4,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
                    )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', losses.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x1', losses_x1.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x2', losses_x2.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x3', losses_x3.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x4', losses_x4.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)
            
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
