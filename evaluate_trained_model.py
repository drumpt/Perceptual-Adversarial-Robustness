
from typing import Dict, List
import math
import torch
import csv
import argparse
from datetime import datetime

from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model
from perceptual_advex.attacks import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial training evaluation')

    add_dataset_model_arguments(parser, include_checkpoint=True)
    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of GPUs to train on')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--per_example', action='store_true', default=False,
                        help='output per-example accuracy')
    parser.add_argument('--output', type=str, help='output CSV')

    args = parser.parse_args()

    dataset, model = get_dataset_model(args)
    # train_loader, val_loader = dataset.make_loaders(1, args.batch_size, only_val=True)
    train_loader, val_loader = dataset.make_loaders(1, args.batch_size, shuffle_val=False) # fix validation data order

    # print(len(train_loader))
    # print(len(val_loader))

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # # for using adversarial-robustness-toolbox(art)
    # inputs, labels = next(iter(train_loader))
    # input_shape = inputs.shape[1:]
    # nb_classes = int(model(inputs.cuda()).shape[1])

    attack_names: List[str] = args.attacks
    attacks = [eval(attack_name) for attack_name in attack_names]

    # Parallelize
    if torch.cuda.is_available():
        # device_ids = list(range(args.parallel))
        # model = nn.DataParallel(model, device_ids)
        # attacks = [nn.DataParallel(attack, device_ids) for attack in attacks]
        model = nn.DataParallel(model)
        attacks = [nn.DataParallel(attack) for attack in attacks]

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}

    # for reducing validation size
    total_num_data = 1000
    total_num_batches = math.ceil(total_num_data / len(val_loader))

    for batch_index, (inputs, labels) in enumerate(val_loader):
        if (
            args.num_batches is not None and
            batch_index >= args.num_batches
        ):
            break

        if batch_index >= total_num_batches:
            break

        # # for debugging
        # if batch_index == 0:
        #     print(labels)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        for attack_name, attack in zip(attack_names, attacks):
            adv_inputs = attack(inputs, labels)
            with torch.no_grad():
                adv_logits = model(adv_inputs)
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  sep='\t')
            batches_correct[attack_name].append(batch_correct)

    print('OVERALL')
    accuracies = []
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              sep='\t')
        accuracies.append(accuracy)

    # output_filename = args.output.split(".")[0] + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv"
    output_filename = args.output
    with open(output_filename, 'w') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow(attack_names)
        if args.per_example:
            for example_correct in zip(*[
                attacks_correct[attack_name] for attack_name in attack_names
            ]):
                out_csv.writerow(
                    [int(attack_correct.item()) for attack_correct
                     in example_correct])
        out_csv.writerow(accuracies)
