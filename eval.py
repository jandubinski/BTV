import os
import json
import tqdm

import torch
import numpy as np

import utils
from datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier

from datasets.registry import get_dataset

def is_printable(layer_name):
    printable = 'Batch' not in layer_name
    printable &= 'Flatten' not in layer_name
    printable &= 'Normalizer' not in layer_name
    return printable

# def score_me(datas, model, hardware, hardware_worst, stats):

#     reses = {'est':[], 'est_wst':[], 'ratio':[]}

#     hooks = simul.add_hooks(model, stats)

#     for i, dat in enumerate(datas):
#         stats.__reset__()
#         _ = model(dat.unsqueeze(dim=0))
#         energy_est = simul.get_energy_estimate(stats, hardware)
#         reses['est'].append(energy_est)
#         energy_est_worst = simul.get_energy_estimate(stats, hardware_worst)
#         reses['est_wst'].append(energy_est_worst)
#         rs = energy_est/energy_est_worst
#         reses['ratio'].append(rs)
#         print(f"{i} {rs}", end="\r")
#     print()

#     simul.remove_hooks(hooks)

#     return reses

def eval_single_dataset(image_encoder, dataset_name, args,p, ids='',head_name='default'):
    if head_name == 'default':
        head_name = dataset_name
    else:
        head_name = dataset_name+'Val'
    classification_head = get_classification_head(args, dataset_name,poison=p)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        poison_name=args.attack_method,
        location=args.data_location,
        batch_size=args.batch_size,
        poison=p,
        portion=args.poison_rate,
        trigger_type = args.trigger_type,
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    # res = []
    # res1 = []
    
 
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(dataloader, disable=True)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            image_encoder = image_encoder.to(device)
            features = image_encoder(x)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            # res.extend(features.detach().cpu().numpy())
            # res1.extend(logits.detach().cpu().numpy())
            
            n += y.size(0)


        top1 = correct / n

    metrics = {'top1': top1}
    # res = np.array(res)
    # res1 = np.array(res1)
    # np.save(args.save_res + f"/features_res_{ids}.npy", res)
    # np.save(args.save_res + f"/logits_res_{ids}.npy", res1)
    print('total:',n)
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics

def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info