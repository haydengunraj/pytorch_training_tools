import time
import torch
import numpy as np
from torch.utils import data

from .losses import select_triplets, TripletLoss


class Trainer:
    def __init__(self, model, dataset, optimizer, loss_func, writer=None,
                 batch_size=1, num_workers=1, log_interval=1):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.writer = writer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._log_interval = log_interval
        self._loss_sum = 0

    def train_epoch(self, epoch, device='cpu', start_step=0):
        raise NotImplementedError

    def _log_and_print_loss(self, epoch, step, loss):
        self._loss_sum += loss
        if not step % self._log_interval:
            avg_loss = self._loss_sum/self._log_interval
            if self.writer is not None:
                self.writer.add_scalar('train/loss', avg_loss, step)
            print('[epoch: {}, step: {}, loss: {:.3f}]'.format(epoch, step, avg_loss))
            self._loss_sum = 0


class TripletTrainer(Trainer):
    """Training class which trains one 'epoch' at a time
    Based on FaceNet implementation: https://github.com/davidsandberg/facenet"""
    def __init__(self, model, dataset, optimizer, loss_func, num_classes, images_per_class, epoch_size, writer=None,
                 batch_size=1, num_workers=1, embedding_size=128, classes_per_batch=None, log_interval=1):
        super().__init__(model, dataset, optimizer, loss_func, writer, batch_size, num_workers, log_interval)
        if not isinstance(loss_func, TripletLoss):
            raise ValueError('TripletTrainer may not use {} as a loss function'.format(type(loss_func).__name__))
        if batch_size % 3:
            raise ValueError('batch_size must be a multiple of 3')
        self.num_classes = num_classes
        self.class_indices = list(range(self.num_classes))
        target_array = np.array(self.dataset.targets)
        self.class_samples = [np.where(target_array == i)[0] for i in self.class_indices]
        max_per_class = min((len(samples) for samples in self.class_samples))
        if images_per_class > max_per_class:
            raise ValueError('images_per_class may not exceed the smallest number of examples for a class')
        self.classes_per_batch = classes_per_batch if classes_per_batch is not None else self.num_classes
        if self.classes_per_batch > self.num_classes:
            raise ValueError('classes_per_batch may exceed the number of classes')
        self.images_per_class = images_per_class
        self.images_per_batch = self.classes_per_batch*self.images_per_class
        self.epoch_size = epoch_size
        self.embedding_size = embedding_size
        self.alpha = self.loss_func.alpha

    def train_epoch(self, epoch, device='cpu', start_step=0):
        """Train for one epoch"""
        end_step = start_step + self.epoch_size
        while start_step < end_step:
            # Select examples and set up forward pass dataloader
            batch_indices = self.sample_images()
            forward_subset = data.Subset(self.dataset, batch_indices)
            forward_loader = data.DataLoader(forward_subset, batch_size=self.batch_size,
                                             shuffle=False, num_workers=self.num_workers)

            # Perform forward pass
            print('Starting forward pass...', end='')
            t0 = time.time()
            embeddings = torch.zeros((self.images_per_batch, self.embedding_size))
            with torch.no_grad():
                for batch_num, data_batch in enumerate(forward_loader):
                    batch, _ = data_batch
                    batch = batch.to(device)
                    emb = self.model(batch)
                    embeddings[batch_num*self.batch_size:(batch_num+1)*self.batch_size, :] = emb
            print('done ({:.3f} seconds)'.format(time.time() - t0))

            # Select triplets for training
            print('Selecting triplets for training...', end='')
            t0 = time.time()
            triplets = select_triplets(embeddings, batch_indices, self.classes_per_batch,
                                       self.images_per_class, self.alpha)
            print('done, {} triplets selected ({:.3f} seconds)'.format(len(triplets), time.time() - t0))
            if self.writer is not None:
                self.writer.add_scalar('train/num_triplets', len(triplets), start_step)

            # Set up training dataloader
            triplets = [i for triplet in triplets for i in triplet]  # flatten triplets
            train_subset = data.Subset(self.dataset, triplets)
            train_loader = data.DataLoader(train_subset, batch_size=self.batch_size,
                                           shuffle=False, num_workers=self.num_workers)

            # Train on selected triplets
            for batch, lab in train_loader:
                self.optimizer.zero_grad()

                batch = batch.to(device)
                emb = self.model(batch)

                loss = self.loss_func(emb)
                loss.backward()
                self.optimizer.step()

                start_step += 1
                self._log_and_print_loss(epoch + 1, start_step, loss.item())
                if start_step >= end_step:
                    break
        return start_step

    def sample_images(self):
        sample_indices = []
        if self.classes_per_batch != self.num_classes:
            classes = np.random.choice(self.class_indices, self.classes_per_batch, replace=False)
        else:
            classes = self.class_indices
        for i in classes:
            sample_indices.extend(list(np.random.choice(self.class_samples[i], self.images_per_class, replace=False)))
        return sample_indices


class SoftmaxTrainer(Trainer):
    """Training class which trains one epoch at a time"""
    def __init__(self, model, dataset, optimizer, loss_func, writer=None,
                 batch_size=1, num_workers=1, log_interval=1):
        super().__init__(model, dataset, optimizer, loss_func, writer, batch_size, num_workers, log_interval)

    def train_epoch(self, epoch, device='cpu', start_step=0):
        """Train for one epoch"""
        # Set up dataloader
        train_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=self.num_workers)
        # Train for an epoch
        print('Starting epoch {}'.format(epoch + 1))
        for batch, lab in train_loader:
            self.optimizer.zero_grad()

            batch, lab = batch.to(device), lab.to(device)
            out = self.model(batch)

            loss = self.loss_func(out, lab)
            loss.backward()
            self.optimizer.step()

            start_step += 1
            self._log_and_print_loss(epoch + 1, start_step, loss.item())
        return start_step


class DWSConvTrainer(Trainer):
    """Training class which trains one epoch at a time"""
    def __init__(self, model, dataset, optimizer, loss_func, writer=None,
                 batch_size=1, num_workers=1, log_interval=1):
        super().__init__(model, dataset, optimizer, loss_func, writer, batch_size, num_workers, log_interval)

    def train_epoch(self, epoch, device='cpu', start_step=0):
        """Train for one epoch"""
        # Set up dataloader
        train_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=self.num_workers)
        # Train for an epoch
        print('Starting epoch {}'.format(epoch + 1))
        for batch, lab in train_loader:
            self.optimizer.zero_grad()

            batch, lab = batch.to(device), lab.to(device)
            out = self.model(batch)

            loss = self.loss_func(out, batch)
            loss.backward()
            self.optimizer.step()

            start_step += 1
            self._log_and_print_loss(epoch + 1, start_step, loss.item())
        return start_step
