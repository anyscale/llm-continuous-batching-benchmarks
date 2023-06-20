#!/usr/bin/env python3

import math
from enum import Enum
from transformers import AutoTokenizer
from typing import List
import aiohttp
import argparse
import asyncio
import itertools
import json
import os
import random
import requests
import sys
import time
import numpy as np


def get_wait_time(mean_time_between_requests: float, distribution: str) -> float:
    if distribution == "uniform":
        return mean_time_between_requests
    else:
        return np.random.exponential(mean_time_between_requests)


def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return


async def async_request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                await asyncio.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return


class GenerationBackend(str, Enum):
    HfTextGenerationInference = "HfTextGenerationInference"
    vLLM = "vLLM"
    NaiveHfPipeline = "NaiveHfPipeline"
    RayGen = "RayGen"
    FasterTransformer = "FasterTransformer"


async def query_model_hf(prompt, verbose, tokenizer, allow_variable_generation_length, total_requests, port):
    prompt, prompt_len, response_len = prompt

    timeout = aiohttp.ClientTimeout(total=60*60)

    response_len = max(response_len, 1)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generate_input = dict(
            inputs=prompt,
            parameters=dict(
                max_new_tokens=response_len,
            ),
        )

        if verbose:
            print('Querying model')
        async with session.post(f'http://localhost:{port}/generate', json=generate_input) as resp:
            if verbose:
                print('Done')

            output = await resp.json()
            output['response_len'] = response_len
            if verbose and 'generated_text' in output:
                print(json.dumps(output['generated_text']))

            return (prompt, output)


async def query_model_naive_hf(prompt, verbose, tokenizer, allow_variable_generation_length, total_requests, port):
    prompt, prompt_len, response_len = prompt

    timeout = aiohttp.ClientTimeout(total=6*60*60)

    bs = int(os.environ.get('NAIVE_HF_BS'))

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generate_input = dict(
            total_benchmark_requests=total_requests,
            inputs=prompt,
            parameters=dict(
                batch_size=bs,
                max_length=response_len + prompt_len,
                prompt_len=prompt_len,
                allow_variable_generation_length=allow_variable_generation_length,
                reponse_len=response_len,
            ),
        )

        if verbose:
            print('Querying model')
        async with session.post(f'http://localhost:{port}/generate', json=generate_input) as resp:
            if verbose:
                print('Done')

            output = await resp.json()
            if verbose and 'generated_text' in output:
                print(json.dumps(output['generated_text']))

            output['naive_hf_lens'] = (prompt_len, response_len)
            output['response_len'] = response_len

            return (prompt, output)


async def query_model_ray(prompt, verbose, tokenizer, allow_variable_generation_length, total_requests, port):
    prompt, prompt_len, response_len = prompt

    timeout = aiohttp.ClientTimeout(total=60*60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generate_input = dict(
            total_benchmark_requests=total_requests,
            inputs=prompt,
            parameters=dict(
                prompt_len=prompt_len,
                reponse_len=response_len,
            ),
        )

        if verbose:
            print('Querying model')
        async with session.post(f'http://localhost:{port}/generate', json=generate_input) as resp:
            if verbose:
                print('Done')

            output = await resp.json()
            output['response_len'] = response_len
            if verbose and 'generated_text' in output:
                print(json.dumps(output['generated_text']))
            return (prompt, output)


async def query_model_ft(prompt, verbose, tokenizer, allow_variable_generation_length, total_requests, port):
    prompt, prompt_len, response_len = prompt

    timeout = aiohttp.ClientTimeout(total=4*60*60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generate_input = dict(
            total_benchmark_requests=total_requests,
            inputs=prompt,
            parameters=dict(
                prompt_len=prompt_len,
                reponse_len=response_len,
            ),
        )

        if verbose:
            print('Querying model')
        async with session.post(f'http://localhost:{port}/generate', json=generate_input) as resp:
            if verbose:
                print('Done')

            output = await resp.json()
            output['response_len'] = response_len
            if verbose and 'generated_text' in output:
                print(json.dumps(output['generated_text']))

            num_output_tokens = output["num_output_tokens"]
            num_response_tokens_according_to_ft = num_output_tokens - prompt_len

            print(
                f'ft output comparison, {response_len=} {num_response_tokens_according_to_ft=}')

            # FT returns with prompt. We remove it for accurate metrics.
            if 'generated_text' in output:
                output['generated_text'] = output['generated_text'][len(
                    prompt):]

            return (prompt, output)


async def query_model_vllm(prompt, verbose, tokenizer, allow_variable_generation_length, total_requests, port):
    prompt, prompt_len, expected_response_len = prompt

    timeout = aiohttp.ClientTimeout(total=4*60*60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        generate_input = dict(
            inputs=prompt,
            parameters=dict(
                prompt_len=prompt_len,
                reponse_len=expected_response_len,
            ),
        )

        if verbose:
            print('Querying model')
        async with session.post(f'http://localhost:{port}/generate', json=generate_input) as resp:
            if verbose:
                print('Done')

            output = await resp.json()
            # necessary for latency calc
            output['response_len'] = expected_response_len
            if verbose and 'generated_text' in output:
                print(json.dumps(output['generated_text']))

            return (prompt, output)


def load_prompts(prompt_file):
    with open(prompt_file) as f:
        prompts = [json.loads(l) for l in f.readlines()]
    return prompts


def get_tok_id_lens(tokenizer, batch):
    tokenized = tokenizer.batch_encode_plus(batch)
    lens = [len(s) for s in tokenized['input_ids']]
    # print(lens)
    return lens


def calculate_throughput(queries, dur_s, backend, tokenizer, median_token_latency, median_e2e_latency, all_e2e_latencies, all_per_token_latencies, results_filename, log_latencies, fail_on_response_failure):
    prompts = []
    responses = []
    naive_hf_lens = []
    ft_lens = []
    expected_response_lens = []
    ray_gen_lens = []
    cf_gen_lens = []
    for prompt, response in queries:
        if 'generated_text' in response:
            prompts.append(prompt)
            responses.append(response['generated_text'])
        if 'naive_hf_lens' in response:
            naive_hf_lens.append(response['naive_hf_lens'])
        if 'ray_gen_len' in response:
            ray_gen_lens.append(response['ray_gen_len'])
        if 'num_output_tokens_cf' in response:
            cf_gen_lens.append(response['num_output_tokens_cf'])

        if 'response_len' in response:
            expected_response_lens.append(response['response_len'])

    prompt_ids = [p for p in tokenizer.batch_encode_plus(prompts)['input_ids']]
    response_ids = [r for r in tokenizer.batch_encode_plus(responses)[
        'input_ids']]

    print(
        f'check_len actual {list(sorted(len(response) for response in response_ids))}')
    print(f'check_len expect {list(sorted(expected_response_lens))}')
    print(f'   self-reported {list(sorted(cf_gen_lens))}')

    # for prompt, response, expected_response_len in zip(prompt_ids, response_ids, expected_response_lens):
    #    print(f'check lens {len(prompt)=} {len(response)=} {expected_response_len=}')

    try:
        prompt_lens = get_tok_id_lens(tokenizer, prompts)
        response_lens = get_tok_id_lens(tokenizer, responses)
    except Exception:
        print(prompts)
        print(responses)
        raise

    print(f'naive_hf_lens {list(sorted(naive_hf_lens))}')
    print(f'prompt_lens {list(sorted(prompt_lens))}')
    print(f'calc_throughput response_lens {list(sorted(response_lens))}')
    print(f'expected_response_lens {list(sorted(expected_response_lens))}')
    print(f'ray_gen_lens {list(sorted(ray_gen_lens))}')

    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)

    if naive_hf_lens:
        # Manually count naive hf tok len
        total_resp_tokens = sum(
            [response_len for _, response_len in naive_hf_lens])
        total_prompt_tokens = sum(
            [prompt_len for prompt_len, _ in naive_hf_lens])

        response_token_count = total_prompt_tokens + total_resp_tokens

    if ray_gen_lens:
        response_token_count = sum(ray_gen_lens)

    if backend == GenerationBackend.NaiveHfPipeline:
        # It returns the prompt in the output.
        prompt_token_count = 0

    if backend == GenerationBackend.FasterTransformer:
        response_token_count = sum(expected_response_lens)

    if cf_gen_lens:
        response_token_count = sum(cf_gen_lens)

    # print(f'prompt_token_count {prompt_token_count} response_token_count {response_token_count}')

    throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    # print(f'throughput_tok_s {throughput_tok_s:.02f}')

    qps = len(responses) / dur_s

    with open(results_filename, 'a') as f:
        msg = f'backend {backend} dur_s {dur_s:.02f} tokens_per_s {throughput_tok_s:.02f} qps {qps:.02f} successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}, {median_token_latency=}, {median_e2e_latency=}'
        if log_latencies:
            msg += f' {all_e2e_latencies=} {all_per_token_latencies=}'
        print(msg, file=f)
        print(msg)

    if fail_on_response_failure:
        assert len(responses) == len(
            queries), f"{fail_on_response_failure=}, expected number of successful respones to equal number of queries, got {len(responses)} vs {len(queries)}"


def calculate_cdf(latencies):
    hist, bin_edges = np.histogram(latencies)
    cumsum = np.cumsum(hist)
    print(f"{bin_edges=}")
    print(f"{hist=}")
    print(f"{cumsum=}")


class MeasureLatency:
    def __init__(self):
        self._latencies = []
        self._per_token_latencies = []

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            prompt, output = await f(*args, **kwargs)

            # Do not record latency if request failed.
            if 'generated_text' in output:
                latency = time.time() - start
                self._latencies.append(latency)
                try:
                    self._per_token_latencies.append(
                        latency / output['response_len'])
                except ZeroDivisionError:
                    # Not currently using this metric..
                    pass

            return prompt, output
        return measured


def get_token_ids(input_str, tokenizer):
    t = tokenizer(input_str)
    return t['input_ids']


async def benchmark(
    backend: GenerationBackend,
    tokenizer,
    prompts: List[str],
    allow_variable_generation_length: bool,
    verbose: bool,
    results_filename: str,
    port: int,
    distribution: str,
    qps: float,
    log_latencies: bool,
    fail_on_response_failure: bool,
):

    if backend == GenerationBackend.vLLM:
        query_model = query_model_vllm
    elif backend == GenerationBackend.HfTextGenerationInference:
        query_model = query_model_hf
    elif backend == GenerationBackend.NaiveHfPipeline:
        query_model = query_model_naive_hf
    elif backend == GenerationBackend.RayGen:
        query_model = query_model_ray
    elif backend == GenerationBackend.FasterTransformer:
        query_model = query_model_ft
    else:
        raise ValueError(f'unknown backend {backend}')

    m = MeasureLatency()

    query_model = m.measure(query_model)

    if distribution == "burst":
        qps = float('inf')

    print(
        f'Starting with backend={backend}, num_prompts={len(prompts)}, allow_variable_generation_length={allow_variable_generation_length}')
    print(f'traffic distribution={distribution}, qps={qps}')

    total_requests = len(prompts)

    async_prompts = async_request_gen(
        iter(prompts), qps=qps, distribution=distribution)

    start_time = time.time()
    tasks = []
    async for prompt in async_prompts:
        tasks.append(asyncio.create_task(query_model(
            prompt, verbose, tokenizer, allow_variable_generation_length, total_requests, port)))
    queries = await asyncio.gather(*tasks)
    dur_s = time.time() - start_time

    median_token_latency = np.median(m._per_token_latencies)
    median_e2e_latency = np.median(m._latencies)

    calculate_throughput(queries, dur_s, backend, tokenizer, median_token_latency, median_e2e_latency,
                         m._latencies, m._per_token_latencies, results_filename, log_latencies, fail_on_response_failure)
    calculate_cdf(m._latencies)


def gen_random_response_lens(distribution: str, len_mean, len_range, num_prompts):
    if distribution == 'uniform':
        if len_range == 0:
            return [len_mean for _ in range(num_prompts)]

        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        num_to_generate = list(
            map(lambda _: random.randint(low, high), range(num_prompts)))
        return num_to_generate
    elif distribution == 'exponential':
        np.random.seed(random.randint(0, 1e6))
        return [min(round(s), len_range) for s in np.random.exponential(scale=len_mean, size=num_prompts)]
    elif distribution == 'capped_exponential':
        np.random.seed(random.randint(0, 1e6))
        response_lens = []
        while len(response_lens) < num_prompts:
            sample = round(np.random.exponential(scale=len_mean))
            if sample <= len_range:
                response_lens.append(sample)
        return response_lens
    else:
        raise ValueError(f'unknown distribution {distribution=}')


def gen_random_prompts(tokenizer, len_mean, len_range, num_prompts, vocab_ids_to_exclude=[]):
    prompts, _ = gen_random_prompts_return_lens(
        tokenizer, len_mean, len_range, num_prompts, vocab_ids_to_exclude)
    return prompts


def gen_random_prompts_return_lens(tokenizer, len_mean, len_range, num_prompts, vocab_ids_to_exclude=[]):

    low = len_mean - (len_range // 2)
    high = len_mean + (len_range // 2)
    vocab_ids = list(set(tokenizer.get_vocab().values()) -
                     set(vocab_ids_to_exclude))

    def gen_prompt_ids(length):
        return [random.randint(10, 50000) for _ in range(length)]

    prompt_lens = list(
        map(lambda _: random.randint(low, high), range(num_prompts)))
    prompts_as_ids = list(
        map(lambda prompt_len: gen_prompt_ids(prompt_len), prompt_lens))
    prompts = list(
        map(lambda prompt_ids: tokenizer.decode(prompt_ids), prompts_as_ids))

    # Because tokens do not map 1:1 to words, sometimes we get more tokens than desired.
    # This removes the additional tokens by tokenizing the prompt and cutting off additional tokens.
    # Confusingly, it works with a single iteration per prompt.
    for i, (p, l) in enumerate(zip(prompts, prompt_lens)):
        encoded = tokenizer(p)['input_ids']
        if len(encoded) > l:
            # I am not sure why l-1 works, but it does..
            encoded = encoded[:l - 1]
        decoded = tokenizer.decode(encoded)
        encoded = tokenizer(decoded)['input_ids']
        assert len(
            encoded) == l, f"Expected prompt to contain exactly {l} tokens, got {len(encoded)=}"
        prompts[i] = decoded

    return prompts, prompt_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--backend', type=GenerationBackend,
                        choices=[e.name for e in GenerationBackend], required=True)
    parser.add_argument('--results_filename', type=str, default='log')
    parser.add_argument('--port', type=int, required=True)

    parser.add_argument('--random_prompt_lens_mean', type=int)
    parser.add_argument('--random_prompt_lens_range', type=int)
    parser.add_argument('--random_prompt_count', type=int)

    parser.add_argument(
        '--distribution', choices=["burst", "uniform", "poisson"], default="burst")
    parser.add_argument('--qps', type=float, default=5.0)

    parser.add_argument('--log_latencies', action="store_true",
                        help="Whether or not to write all latencies to the log file.")
    parser.add_argument('--fail_on_response_failure', action="store_true",
                        help="Whether or not to fail the benchmarking script if any request fails")

    parser.add_argument('--variable_response_lens_mean', type=int)
    parser.add_argument('--variable_response_lens_range', type=int)
    parser.add_argument('--variable_response_lens_distribution', choices=[
                        "uniform", "exponential", "capped_exponential"], default="uniform")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompts_filename', type=str)
    group.add_argument('--gen_random_prompts', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--allow_variable_generation_length',
                       action='store_true')
    group.add_argument('--fixed_max_tokens', type=int)

    parser.add_argument('--print-generation-lens-and-exit',
                        action='store_true')

    args = parser.parse_args()

    if args.gen_random_prompts:
        assert args.random_prompt_count is not None

    backend = GenerationBackend[args.backend]
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-13b')

    if args.prompts_filename:
        prompts = load_prompts(args.prompts_filename)
        prompt_lens = itertools.repeat(-1)
        num_prompts = len(prompts)
    elif args.gen_random_prompts:
        num_prompts = args.random_prompt_count
        random.seed(0xCADE)
        prompts, prompt_lens = gen_random_prompts_return_lens(
            tokenizer,
            len_mean=args.random_prompt_lens_mean,
            len_range=args.random_prompt_lens_range,
            num_prompts=num_prompts,
            vocab_ids_to_exclude=tokenizer.all_special_ids,
        )
    else:
        raise ValueError("unknown prompts")

    if args.allow_variable_generation_length:
        response_lens = gen_random_response_lens(
            args.variable_response_lens_distribution, args.variable_response_lens_mean, args.variable_response_lens_range, num_prompts=num_prompts)
        args.fixed_max_tokens = -1
    else:
        response_lens = [args.fixed_max_tokens for _ in range(num_prompts)]

    for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
        total = prompt_len + gen_len
        if total > 2048:
            print(f'truncating long prompt+gen_len {prompt_len=} {gen_len=}')
            gen_len = 2048 - prompt_len
        response_lens[i] = gen_len

    if args.print_generation_lens_and_exit:
        print(f'{prompt_lens=}')
        print(f'{response_lens=}')
        print('Exiting...')
        return

    if args.verbose or True:
        print('prompt lens', prompt_lens)
        print('response lens', response_lens)

        totals = []
        for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
            totals.append(prompt_len + gen_len)

        print('total tokens', list(sorted(totals)))

    prompts = list(zip(prompts, prompt_lens, response_lens))

    asyncio.run(benchmark(
        backend,
        tokenizer,
        prompts,
        args.allow_variable_generation_length,
        args.verbose,
        args.results_filename,
        args.port,
        args.distribution,
        args.qps,
        args.log_latencies,
        args.fail_on_response_failure,
    ))


if __name__ == '__main__':
    main()
