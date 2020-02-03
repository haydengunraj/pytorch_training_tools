from cnn.training_setup import initialize_conversion_experiment, get_lr

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str, help='Name of experiment')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from most recent checkpoint')
    parser.add_argument('--reset_optim', action='store_true', help='Flag to reset optimizer on resume')
    parser.add_argument('--reset_sched', action='store_true', help='Flag to reset scheduler on resume')
    args = parser.parse_args()

    (model, train_dataset, val_dataset,
     trainer, evaluator, scheduler,
     loss_func, optimizer, saver, writer,
     device, start_epoch, start_step) = initialize_conversion_experiment(args.experiment, args.resume,
                                                                         args.reset_optim, args.reset_sched)

    # Train model
    step = start_step
    model.train()
    val_metrics = None
    for epoch in range(start_epoch, args.epochs):
        # Log learning rate
        if writer is not None:
            writer.add_scalar('train/learning_rate', get_lr(optimizer), step)

        # Train one epoch
        step = trainer.train_epoch(epoch, device=device, start_step=step)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Evaluate model
        if evaluator is not None:
            val_metrics = evaluator.eval(step, device=device)

        # Save checkpoint
        if saver is not None:
            saver.update_checkpoint(epoch + 1, step, metric_vals=val_metrics)

        # Flush remaining events to disk
        if writer is not None:
            writer.flush()