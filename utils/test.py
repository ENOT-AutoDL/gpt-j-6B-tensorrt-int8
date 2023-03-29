from typing import Callable

import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def test_acc(predict_last_id_function: Callable[[torch.Tensor], torch.Tensor], verbose: bool = True) -> float:
    """
    Calculates accuracy of the model (lambada dataset).

    Parameters
    ----------
    predict_last_id_function : Callable[[torch.Tensor], torch.Tensor]
        Callable object, which returns next token id predicted by the model.
    verbose : bool
        Shows the progress of calculations, if the parameter is True.

    Returns
    -------
    float
        Model accuracy.

    """

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')

    dataset = load_dataset('lambada', split='validation')
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(lambda data: tokenizer(data['text']), batched=True)
    dataset.set_format(type='torch', columns=['input_ids'])

    total, hit = 0, 0
    pbar = tqdm(dataset, disable=not verbose)
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].unsqueeze(0)

        gt_id = input_ids[:, -1]
        pred_id = predict_last_id_function(input_ids[:, :-1])

        total += gt_id.size(0)
        hit += (pred_id == gt_id).sum().item()

        pbar.set_description(f'Acc = {100.0 * (hit / total):.2f}%')

    final_acc = (hit / total) * 100
    if verbose:
        print(f'Final acc = {final_acc:.2f}%')

    return final_acc
