import os
import torch
import argparse


def _export(state_dict, component, destination, stem):
    torch.save(state_dict[component], os.path.join(destination, stem + '_' + component + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_file', type=str, help='Checkpoint file to convert (.pth file)')
    parser.add_argument('destination', type=str, help='Directory to save exported weights to')
    parser.add_argument('--export_optim', action='store_true', help='Flag for exporting optimizer state dict')
    parser.add_argument('--export_sched', action='store_true', help='Flag for exporting scheduler state dict')
    args = parser.parse_args()

    # Set up destination and get file stem
    os.makedirs(args.destination, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint_file))[0]
    state_dict = torch.load(args.checkpoint_file)

    # Export individual state dicts
    _export(state_dict, 'model', args.destination, stem)
    if args.export_optim:
        _export(state_dict, 'optimizer', args.destination, stem)
    if args.export_sched and state_dict['scheduler'] is not None:
        _export(state_dict, 'scheduler', args.destination, stem)
    print('Exported checkpoint to', args.destination)
