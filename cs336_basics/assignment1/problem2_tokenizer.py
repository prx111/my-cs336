import json
import regex as re
from collections.abc import Iterable, Iterator
from functools import partial
from tqdm import tqdm
import os
import time
import io


class BPETokenizer:
    def __init__(self, vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]], 
                special_tokens:list[str] | None=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.vocab_length = len(vocab)
        self.token_to_index = {token: index for index, token in vocab.items()}
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}
        if self.special_tokens:
            self.vocab.update({i+self.vocab_length: token.encode('utf-8') for i, token in enumerate(self.special_tokens)})
        
        self.pretokenize_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.special_pattern = (
            re.compile("|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            if self.special_tokens else None
        )
    
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, 
                   special_tokens:list[str] | None=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        """
        
        vocab = {}  
        if vocab_filepath.endswith('.json'):    
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        elif vocab_filepath.endswith('.txt'):
            with open(vocab_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # split only on the first tab so that tokens containing tabs are preserved
                    parts = line.rstrip("\n").split('\t', 1)
                    if len(parts) < 2:
                        continue
                    index_str, token = parts
                    vocab[int(index_str)] = token.encode('utf-8')
        else:
            raise ValueError("Unsupported vocab file format. Supported formats are .json and .txt")
        
        merges = []
        with open(merges_filepath, 'rb') as f:
            for line in f:
                parts = line.rstrip(b'\n').split(b'\t')
                if len(parts) < 2:
                    continue
                token1, token2 = parts
                merges.append((token1, token2))
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs"""

        # 1. pretokenize the input text into a list of bytes(split the special tokens first)
        if self.special_tokens:
            spec_tokens = [match.group() for match in self.special_pattern.finditer(text)]
            segments = self.special_pattern.split(text)
        else:
            segments = [text]

        final_result = []

        for i, segment in enumerate(segments):
            pretokenize_byte_ls = [
                match.group().encode('utf-8') for match in self.pretokenize_pattern.finditer(segment)
            ]

            # 2.apply the merges
            result = [] if i == 0 else [spec_tokens[i-1].encode('utf-8')]
            for word in pretokenize_byte_ls:
                byte_token = [bytes([item]) for item in word]
                while True:
                    
                    merge_pos =  -1
                    pair_index = self.vocab_length
                    
                    for i in range(len(byte_token) - 1):
                        pair = (byte_token[i], byte_token[i+1])
                        if pair not in self.merges_dict:
                            continue
                        pair_pos = self.merges_dict[pair]
                        if pair_pos < pair_index:
                            pair_index = pair_pos
                            merge_pos = i
                            merge_pair = pair[0] + pair[1]
                    
                    if merge_pos != -1:
                        suffix = [merge_pair] + byte_token[merge_pos + 2:] if merge_pos < len(byte_token) - 2 else [merge_pair]
                        byte_token = byte_token[:merge_pos] + suffix
                    else:
                        break
                result.extend(byte_token)
            final_result.extend(result)
        
        # 3. convert the list of bytes to a list of token indices
        indices = [self.token_to_index.get(token, 0) for token in final_result]
        return indices
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """

        # ref1:https://lovemma.github.io/posts/python-%E5%A6%82%E4%BD%95%E6%B5%81%E5%BC%8F%E8%AF%BB%E5%8F%96%E5%A4%A7%E6%96%87%E4%BB%B6/
        
        # 1. break the iterable into chunks and process each chunk separately
        block_size = 1024 * 1024 * 1     # 10MB
        num_chunks = (iterable.seek(0, os.SEEK_END) + block_size - 1) // block_size 
        iterable.seek(0) 
        throughput = 0
        
        pbar = tqdm(iter(partial(iterable.read, block_size), ''), total=num_chunks)
        for chunk in pbar:  # '': flag to stop iteration when read() returns an empty string
            start_time = time.time()
            token_ids = self.encode(chunk)
            # compute the throughput of the tokenizer(e.g., bytes/second)
            # throughput: 吞吐量，【处理的字节数】与花费的时间的比值。目前：约1～2 MB /s; 理想：>10 MB /s
            throughput = (len(chunk) / (1024 * 1024)) / (time.time() - start_time)
            pbar.set_description(f"Tokenizing, speed={throughput:.2f} MB/second")
            for token_id in token_ids:       # Yield each token ID individually
                yield token_id

        return

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text,
        In the case that the input token IDs do not produce a valid Unicode string,
        using errors='replace' to replace invalid bytes with the Unicode replacement character (U+FFFD)
        """
        return b"".join(self.vocab.get(i, b"") for i in ids).decode("utf-8", errors='replace')


if __name__ == '__main__':
    
    # # 1.test from_files
    # vocab_filepath = './bpe_outputs/1.TinyStories_vocab.json'
    # merges_filepath = './bpe_outputs/1.TinyStories_merges.txt'
    
    vocab_filepath = './bpe_outputs/2.owt_vocab.json'
    merges_filepath = './bpe_outputs/2.owt_merges.txt'
    tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>"])
    
    with open("../data/owt_train.txt") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)        # 计算corpus的总字节数

        chunk_size = file_size // 10
        print(f"子文档大小:{chunk_size / (1024*1024)}MB")

        for i in range(10):
            ids = []
            print(f"按顺序处理第{i+1}个documents")

            # file.seek(i*chunk_size)
            start = i*chunk_size
            file.seek(start)
            end = min(file_size, start + chunk_size)
            chunk = file.read(end - start)
            text_stream = io.StringIO(chunk)

            for _id in tokenizer.encode_iterable(text_stream):
                ids.append(_id)

            compression_ratio = chunk_size / len(ids)
            print(f"compression_ratio:{compression_ratio}")

        


    # # 2.test encode
    # test_str = "hello, world, fuck, shit."
    # token_ids = tokenizer.encode(test_str)
    # print(token_ids)

    # # 3.test encode_iterable
    # input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    # with open(input_path, "rb") as f:
    #     generator = tokenizer.encode_iterable(f)
    #     for token_ids in generator:
    #         print(token_ids)
    #         break
    

    # # 4. test decode
    # strs = tokenizer.decode(token_ids)
    # print(strs)
    