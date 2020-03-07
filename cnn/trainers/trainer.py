from torch.utils.data import DataLoader


def get_trainer(config):
    return Trainer(**config)


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

    def train_epoch(self, epoch, step, device='cpu'):
        """Train for one epoch"""
        # Set up dataloader
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        # Train for an epoch
        print('Starting epoch {}'.format(epoch))
        for data_batch in train_loader:
            self.optimizer.zero_grad()
            data_dict = {key: val.to(device) for key, val in data_batch.items()}
            data_dict.update(self.model(data_dict))
            data_dict.update(self.loss_func(data_dict))

            data_dict['loss'].backward()
            self.optimizer.step()
            step += 1
            self._log_and_print_loss(epoch, step, data_dict['loss'].item())
        return step

    def _log_and_print_loss(self, epoch, step, loss):
        self._loss_sum += loss
        if not step % self._log_interval:
            avg_loss = self._loss_sum/self._log_interval
            if self.writer is not None:
                self.writer.add_scalar('train/loss', avg_loss, step)
            print('[epoch: {}, step: {}, loss: {:.3f}]'.format(epoch, step, avg_loss), flush=True)
            self._loss_sum = 0
