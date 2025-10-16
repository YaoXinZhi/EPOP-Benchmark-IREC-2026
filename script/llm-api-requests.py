# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 08/01/2024 20:04
@Author: yao
"""

import json
import os
import time
import argparse
import logging
from datetime import datetime

from openai import OpenAI


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REPEAT_TIME = 5


def read_doc(sent_file: str) -> str:
    try:
        with open(sent_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Failed to read file '{sent_file}': {e}")
        raise


def get_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='text_file',
                        default='/Users/yao/Nutstore Files/Mac2PC/LLM-EPOP-RE_BLAH9_2025-01-15/zero-shot-evaluation-main/experiments/documents/Texte2.txt')
    parser.add_argument('-p', dest='prompt_file',
                        default='/Users/yao/Nutstore Files/Mac2PC/LLM-EPOP-RE_BLAH9_2025-01-15/zero-shot-evaluation-main/experiments/instructions/short.txt')
    parser.add_argument('-s', dest='save_path',
                        default='/Users/yao/Nutstore Files/Mac2PC/LLM-EPOP-RE_BLAH9_2025-01-15/zero-shot-evaluation-main/experiments/output/repetition/kimi')
    args = parser.parse_args()

    for path in [args.text_file, args.prompt_file, args.save_path]:
        if not os.path.exists(path):
            logging.warning(f"Path does not exist: {path}")

    return args


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY", "-")
    if api_key == "-":
        logging.warning("API key is not set. Please set DEEPSEEK_API_KEY in environment variables.")

    model = "deepseek-chat"

    args = get_para()
    sent_file = args.text_file
    prompt_file = args.prompt_file
    save_path = args.save_path

    save_dir = os.path.basename(prompt_file).split('.')[0]
    text_prefix = os.path.basename(sent_file).split('.')[0]

    dir_path = os.path.join(save_path, save_dir, text_prefix)
    os.makedirs(dir_path, exist_ok=True)

    logging.info(f'---------model: {model}---------')

    prompt = read_doc(prompt_file)
    sent = read_doc(sent_file)

    logging.info(f'processing {os.path.basename(sent_file)} & {os.path.basename(prompt_file)}.')

    start_time = time.time()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    for repeat in range(1, REPEAT_TIME + 1):
        logging.info(f'repeating-{repeat}.')
        save_file = os.path.join(dir_path, f'{repeat}.txt')

        if os.path.exists(save_file):
            logging.info(f'"{save_file}" exists.')
            continue

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": sent}
                ]
            )

            result = stream.choices[0].message.content if stream.choices else ""
            if not result:
                logging.warning("Received empty response from API.")
                continue

            with open(save_file, 'w', encoding='utf-8') as wf:
                wf.write(f'{result}\n\n')

            logging.info(f'{save_file} saved.')

        except Exception as e:
            logging.error(f"Error during API call or file write: {e}")
            continue

        if repeat < REPEAT_TIME:
            time.sleep(30)

    end_time = time.time()
    logging.info(f'time cost: {end_time - start_time:.4f}s.')


if __name__ == '__main__':
    main()
