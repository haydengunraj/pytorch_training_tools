import time
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

from ..utils import LOSS_KEY
from .trainer import Trainer
from ..losses.triplet import TripletLoss


class TripletTrainer(Trainer):
    """Training class which trains one 'epoch' at a time
    Based on FaceNet implementation: https://github.com/davidsandberg/facenet"""
    def __init__(self, model, dataset, optimizer, loss_func, num_classes, images_per_class,
                 epoch_size, embedding_key, writer=None, embedding_size=128, classes_per_batch=None,
                 batch_size=1, num_workers=1, log_interval=1, tag_prefix='train/'):

        super().__init__(model, dataset, optimizer, loss_func, writer,
                         batch_size, num_workers, log_interval, tag_prefix)
        if not isinstance(loss_func.loss_func, TripletLoss):
            raise ValueError('TripletTrainer may not use {} as a loss function'.format(
                type(loss_func.loss_func).__name__))
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
            raise ValueError('classes_per_batch may not exceed the number of classes')
        self.images_per_class = images_per_class
        self.images_per_batch = self.classes_per_batch*self.images_per_class
        self.epoch_size = epoch_size
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key
        self.alpha = self.loss_func.alpha

    def train_epoch(self, epoch, step, device='cpu'):
        """Train for one epoch"""
        end_step = step + self.epoch_size
        while step < end_step:
            # Select examples and set up forward pass dataloader
            batch_indices = self.sample_images()
            forward_subset = Subset(self.dataset, batch_indices)
            forward_loader = DataLoader(forward_subset, batch_size=self.batch_size,
                                        shuffle=False, num_workers=self.num_workers)

            # Perform forward pass
            print('Starting forward pass...', end='')
            t0 = time.time()
            embeddings = torch.zeros((self.images_per_batch, self.embedding_size))
            with torch.no_grad():
                for batch_num, data_batch in enumerate(forward_loader):
                    data_dict = {key: val.to(device) for key, val in data_batch.items()}
                    data_dict.update(self.model(data_dict))
                    embeddings[batch_num*self.batch_size:(batch_num+1)*self.batch_size, :] = \
                        data_dict[self.embedding_key]
            print('done ({:.3f} seconds)'.format(time.time() - t0))

            # Select triplets for training
            print('Selecting triplets for training...', end='')
            t0 = time.time()
            triplets = self.select_triplets(embeddings, batch_indices)
            print('done, {} triplets selected ({:.3f} seconds)'.format(len(triplets), time.time() - t0))
            if self.writer is not None:
                self.writer.add_scalar(self.tag_prefix + 'num_triplets', len(triplets), step)

            # Set up training dataloader
            triplets = [i for triplet in triplets for i in triplet]  # flatten triplets
            train_subset = Subset(self.dataset, triplets)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=self.num_workers)

            # Train on selected triplets
            for data_batch in train_loader:
                self.optimizer.zero_grad()

                data_dict = {key: val.to(device) for key, val in data_batch.items()}
                data_dict.update(self.model(data_dict))
                data_dict.update(self.loss_func(data_dict))

                data_dict[LOSS_KEY].backward()
                self.optimizer.step()

                step += 1
                self._log_and_print(epoch + 1, step, data_dict)
                if step >= end_step:
                    break
        return step

    def sample_images(self):
        sample_indices = []
        if self.classes_per_batch != self.num_classes:
            classes = np.random.choice(self.class_indices, self.classes_per_batch, replace=False)
        else:
            classes = self.class_indices
        for i in classes:
            sample_indices.extend(list(np.random.choice(self.class_samples[i], self.images_per_class, replace=False)))
        return sample_indices

    def select_triplets(self, embeddings, indices):
        """Randomly selects triplets which violate the triplet loss margin
        Based on FaceNet implementation: https://github.com/davidsandberg/facenet"""
        start_idx = 0
        triplets = []
        for i in range(self.classes_per_batch):
            for j in range(1, self.images_per_class):
                anchor_idx = start_idx + j - 1
                neg_diffs = embeddings[anchor_idx] - embeddings
                neg_dists_sqr = torch.sum(torch.mul(neg_diffs, neg_diffs), 1)
                for k in range(j, self.images_per_class):
                    positive_idx = start_idx + k
                    pos_diff = embeddings[anchor_idx] - embeddings[positive_idx]
                    pos_dist_sqr = torch.sum(torch.mul(pos_diff, pos_diff))
                    neg_dists_sqr[start_idx:start_idx + self.images_per_class] = pos_dist_sqr + 1.
                    all_neg = torch.where(neg_dists_sqr - pos_dist_sqr < self.alpha)[0]
                    nrof_negs = all_neg.size(0)
                    if nrof_negs > 0:
                        rnd_idx = np.random.randint(nrof_negs)
                        negative_idx = int(all_neg[rnd_idx].numpy())
                        triplets.append((indices[anchor_idx], indices[positive_idx], indices[negative_idx]))
            start_idx += self.images_per_class

        np.random.shuffle(triplets)
        return triplets
