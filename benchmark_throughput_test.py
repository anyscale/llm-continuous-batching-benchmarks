#!/usr/bin/env python3

import benchmark_throughput
from transformers import AutoTokenizer
import random
import json
import sys
import pytest

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-6.7b')


def test_gen_random_prompts_deterministic():
    random.seed(0)
    first_run_prompts, first_run_lens = benchmark_throughput.gen_random_prompts_return_lens(
        tokenizer=tokenizer,
        len_mean=10,
        len_range=10,
        num_prompts=10,
    )

    random.seed(0)
    second_run_prompts, second_run_lens = benchmark_throughput.gen_random_prompts_return_lens(
        tokenizer=tokenizer,
        len_mean=10,
        len_range=10,
        num_prompts=10,
    )

    assert first_run_lens == second_run_lens
    assert first_run_prompts == second_run_prompts


def test_gen_random_prompts_correct_len():
    random.seed(0)
    prompts, lens = benchmark_throughput.gen_random_prompts_return_lens(
        tokenizer=tokenizer,
        len_mean=100,
        len_range=100,
        num_prompts=1000,
    )

    for prompt, expected_len in zip(prompts, lens):
        encoded_input_ids = tokenizer(prompt)['input_ids']
        assert len(encoded_input_ids) == expected_len


def test_gen_resp_lens_deterministic():
    for distribution in ['uniform', 'exponential']:
        iters = []
        for _ in range(2):
            random.seed(0)
            lens = benchmark_throughput.gen_random_response_lens(
                distribution='uniform',
                len_mean=100,
                len_range=100,
                num_prompts=100,
            )
            iters.append(lens)

        assert iters[0] == iters[1], distribution


def test_gen_resp_lens_exponential():
    for distribution in ['exponential', 'capped_exponential']:
        random.seed(0)
        lens = benchmark_throughput.gen_random_response_lens(
            distribution='exponential',
            len_mean=100,
            len_range=1024,
            num_prompts=1000,
        )

        assert max(lens) <= 1024


if __name__ == '__main__':
    sys.exit(pytest.main(["-v", __file__]))
