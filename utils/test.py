import random
import time
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def test_acc(predict_next_id_function: Callable[[torch.Tensor], torch.Tensor], verbose: bool = True) -> float:
    """
    Calculates accuracy of the model (lambada dataset).

    Parameters
    ----------
    predict_next_id_function : Callable[[torch.Tensor], torch.Tensor]
        Callable object, which returns next token id predicted by the model.
    verbose : bool
        Shows the progress of calculations.

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
        pred_id = predict_next_id_function(input_ids[:, :-1])

        total += gt_id.size(0)
        hit += (pred_id == gt_id).sum().item()

        pbar.set_description(f'Acc = {100.0 * (hit / total):.2f}%')

    final_acc = (hit / total) * 100
    if verbose:
        print(f'Final acc = {final_acc:.2f}%')

    return final_acc


def test_latency(
    generate_ids_function: Callable[[int], torch.Tensor],
    generate_seq_function: Callable[[torch.Tensor, int], Any],
    variants: List[Tuple[int, int]],
    warmup: int = 20,
    repeats: int = 20,
    verbose: bool = True,
) -> List[Tuple[int, int, float, float]]:
    """
    Сalculates the latency of the model depending on the length of the input sequence
    and the length of the output sequence.

    Parameters
    ----------
    generate_ids_function : Callable[[int], torch.Tensor]
        Callable object, which returns inputs ids tensor with appropriate data
        type and device.
    generate_seq_function : Callable[[torch.Tensor, int], torch.Tensor]
        Callable object, which runs generation procedure. Must take 2 parameters.
        The first parameter is the input ids tensor. The second parameter is the
        length of the generated sequence.
    variants : List[Tuple[int, int]]
        List of variants for test. Each variant is pair of integer values, where
        the first value is start sequence length and second value is number of
        ids to generate.
    gen_len_variants : List[int]
        List of variants of number of ids to generate.
    warmup : int
        Number of warmup runs. Every warmup run selects input ids length and
        number of ids to generate from test variants randomly.
    repeates : int
        Number of runs for every test variant.
    verbose : bool
        Prints latency table.

    Returns
    -------
    List[Tuple[int, int, float, float]]
        Return latency table as list of 4 values: start sequence length, number of
        generated ids, latency in ms, latency std.

    """

    if verbose:
        print(f'WARMUP = {warmup}')

    for _ in range(warmup):
        seq_len, gen_len = random.choice(variants)
        input_ids = generate_ids_function(seq_len)
        generate_seq_function(input_ids, gen_len)

    if verbose:
        print(f'REPEATS = {repeats}')
        print()
        print(f'SEQUENCE_LENGTH  GENERATE_LENGTH  TIME(ms)  TIME_STD(ms)')

    result = []
    for seq_len, gen_len in variants:

        def calc_time():
            t_0 = time.perf_counter()
            generate_seq_function(input_ids, gen_len)
            t_1 = time.perf_counter()

            return t_1 - t_0

        stats = [calc_time() for _ in range(repeats)]
        mean_ms, std_ms = np.mean(stats) * 1000, np.std(stats) * 1000
        result.append((seq_len, gen_len, mean_ms, std_ms))

        if verbose:
            print(f'{seq_len:4d} {gen_len:4d} {mean_ms:5.0f} {std_ms:4.1f}')

    return result
