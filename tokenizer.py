
import logging
import json
import os
import regex as re

import numpy as np

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(tokens, pair, value):
    processed = []

    i = 0
    while i != len(tokens):
        move_single = True
        
        if i < len(tokens) - 1:
            current_key = (tokens[i], tokens[i + 1])
            if current_key == pair:
                processed.append(value)
                i += 2
                move_single = False
                
        if move_single:
            processed.append(tokens[i])
            i += 1
                
    return processed

class Tokenizer:
    def __init__(self, filename):
        self.pattern_kind = None
        self.compiled_pattern = None
        self.setup_pattern(1)

        self.vocabulary_size = 0
        self.encode_map = dict()
        self.decode_map = dict()

        if os.path.exists(filename):
            self.load(filename)

    def setup_pattern(self, kind):
        self.pattern_kind = kind
        pattern = GPT2_SPLIT_PATTERN if self.pattern_kind == 0 else GPT4_SPLIT_PATTERN 
        self.compiled_pattern = re.compile(pattern)

    def load(self, data_filename):
        self.vocabulary_size = 0
        self.encode_map = dict()
        self.decode_map = dict()
        if not os.path.exists(data_filename):
            logging.error(f"File {data_filename} does not exist")
            return
        
        try:
            with open(data_filename, 'r') as f:
                state = json.load(f)
                self.vocabulary_size = state["size"]

                records = sorted(state["merges"], key=lambda x: x['t'])
                for c in records:
                    k = (c["p0"], c["p1"])
                    v = c["t"]
                    self.encode_map[k] = v

                self.decode_map = {int(k): v for k,v in state["vocab"].items()}

                self.setup_pattern(state["pattern_kind"])

        except Exception as e:
            logging.error(f"Unable to load {data_filename}: {e}")

    def train(self, data_filename, vocabulary_filename, vocabulary_size):
        text = ""
        with open(data_filename, encoding="utf-8") as f:
            text = f.read()

        logging.info(f"Text has {len(text)} characters")

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [ list(map(int, t.encode("utf-8")))  for t in text_chunks]

        start_index = 256
        num_merges = vocabulary_size - start_index

        encode_map = dict()
        decode_map = {idx: [idx] for idx in range(256)} # idx -> bytes
        current_round = 0
        while current_round < num_merges:
            counts = dict()

            # collect the chunk stats
            for chunk_ids in ids:
                counts = get_stats(chunk_ids, counts)

            # find the pair with the highest count
            most_common = max(counts, key=counts.get)

            # create a new id
            new_token = start_index + current_round

            # replace all entries of the token pais
            ids = [merge(chunk_ids, most_common, new_token) for chunk_ids in ids]

            encode_map[most_common] = new_token
            decode_map[new_token] = decode_map[most_common[0]] + decode_map[most_common[1]]  
            logging.info(f"{current_round:03} replaced {most_common} into {new_token}")
            
            current_round = current_round + 1

        state = {
            "size": vocabulary_size,
            "pattern_kind": self.pattern_kind,
            "merges" : [{"p0": p0, "p1": p1, "t": t} for (p0, p1), t in encode_map.items()],
            "vocab": decode_map
        }
        with open(vocabulary_filename, 'w') as f:
            json.dump(state, f, indent=4)

    def replace_id(self, ids, token, pair):
        output = []
        for c in ids:
            if c == token:
                output.extend(pair)
            else:
                output.append(c)
        return output
    

    # 
    # encoding
    #
    def find_next_encoding_token(self, tokens):
        value = None
        pair = None

        iterations = len(tokens) - 1
        if iterations >= 1:
            for i in range(0, iterations):
                key = (tokens[i], tokens[i + 1])
                if key in self.encode_map:
                    token = self.encode_map[key]
                    if value is None or token < value:
                        value = token
                        pair = key

        return pair, value

    def encode(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_ids = chunk.encode("utf-8") # raw bytes
            chunk_ids = list(map(int, chunk_ids)) 

            while True:
                pair, value = self.find_next_encoding_token(chunk_ids)
                if pair is None:
                    break

                #print(f"replacing [{tokens}] {pair} -> {value}")
                chunk_ids = merge(chunk_ids, pair, value)
                
            ids.extend(chunk_ids)
            
        return ids
    
    #
    # decoding
    #
    def decode(self, ids):
        processed = b""
        bucket = b""

        for c in ids:
            if c in self.decode_map:
                #print(f"replacing [{c}] -> {self.decode_map[c]}")
                bucket += bytes(self.decode_map[c])
            else:
                bucket += bytes([c])

            if len(bucket) > 1024:
                processed += bucket
                bucket = b""

        processed += bucket
        return processed.decode("utf-8", errors="replace")
    
    #
    # Helpers
    #
    def debug(self, ids):
        output = ""

        for c in ids:
            if len(output) != 0:
                output += "|"
            output += self.decode([c])
        
        return output
    
    def tokenize_text_file(self, data_file, output_file):
        logging.info(f"Tokenizing {data_file} to {output_file}")

        text = ""
        with open(data_file, encoding="utf-8") as f:
            text = f.read()

        tokens = self.encode(text)
        logging.info(f"Done {len(tokens)} tokens")

        decoded = self.decode(tokens)
        logging.info(f"Decoded {decoded==text}")

        data_np = np.array(tokens, dtype=np.int16)
        data_np.tofile(output_file)
        
