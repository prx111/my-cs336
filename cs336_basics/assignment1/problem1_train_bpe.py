from typing import BinaryIO
import os
import regex as re
from collections import Counter, defaultdict

from collections.abc import Iterable
from typing import IO, Any, BinaryIO

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

import json

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)        # 计算corpus的总字节数

    chunk_size = file_size // desired_num_chunks        # 每个chunk的字节数，共分为n个chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]      # 每个chunk的起始位置
    chunk_boundaries[-1] = file_size    # 列表的最后一个元素为 corpus 最后一个字节 的位置 

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time 

    for bi in range(1, len(chunk_boundaries) - 1):      # 从第一个chunk开始 往后 遍历 直到找到 b'endoftext'
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at      # endoftext的前一个字节 所在的 位置
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(args):
    
    chunk = args
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    pattern = re.escape("|".join(["<|endoftext|>"]))
    segments = re.split(pattern, chunk)

    local_counter = Counter()
    for segment in segments:
        pretokenize = re.finditer(pat, segment)
        tokens = [match.group() for match in pretokenize]
        local_counter.update(tokens)

    return local_counter
    
    

def process_file(input_path, num_processes=4):
    """
    多进程处理chunk.
    注意：
    1) num_processes 和 desired_num_chunks 在面对大数据集时需要不断调整，从而进行内存和效率的权衡。
    2) 每次局部处理 和 返回最终的结果 均适用counter, 达到内存友好的目的。
    """
    chunks = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=1000, split_special_token=b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    
    with Pool(processes=num_processes) as pool:    
        counters = tqdm(pool.imap_unordered(process_chunk, chunks), 
                        total=len(chunks), 
                        desc="Processing chunks")
    
        total_counter = Counter()
        for counter in counters:
            total_counter.update(counter)  # 高效合并
      
    return total_counter

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """ 
    计算邻接字节对的频率
    思路：
    1. 先统计每个单词的频率，得到 word_level_counts
    2. 遍历 word_level_counts，统计每个单词的字节表示，得到 word_to_bytes，同时统计每个字节对的频率，得到 byte_pair_counts
    3. 初始化 vocab，包含256个单字节和特殊token
    4. 开始合并，每次从 byte_pair_counts找到频率最高的字节对；
    5. 遍历所有单词，查看该单词是否存在当前匹配的字节对.
        如果存在，则进行合并，并更新该单词的字节表示，同时更新 byte_pair_counts 中相关字节对的频率
        扣除一部分字节对，更新一部分字节对。
    6. byte_pair_counts中去掉合并的字节对，继续下一轮合并，直到达到 vocab_size

    变量说明：
    byte_pair_counts: key为元祖表示字节对，如(80, 81), value为该字节对在所有单词中出现的频率
    word_to_bytes: key为单词，value为该单词的字节表示，如 "hello" -> [104, 101, 108, 108, 111]
        byte_ls: 字节表示，如[104, 101, 108, 108, 111]
        new_byte_ls: 合并后的字节表示，如[104, 101, 283, 111]，其中283表示合并了108和108的字节对
    bytes_to_word: 倒排索引，只处理受影响的词（即该词中包含字节对). 这里采用set，避免重复添加单词。

    返回：
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    
    """
    
    special_tokens = [token.encode('utf-8') for token in special_tokens]
    word_level_counts = process_file(input_path)
    
    byte_pair_counts = {}
    word_to_bytes = {}
    bytes_to_word = defaultdict(set)    # 倒排索引

    for word, value in tqdm(word_level_counts.items(), total=len(word_level_counts), desc="分别统计字节对频率、词到字节映射、倒排索引"):   
        byte_ls = list(word.encode('utf-8'))
        word_to_bytes[word] = byte_ls
        for i in range(len(byte_ls) - 1):
            pair = tuple(byte_ls[i:i+2])
            byte_pair_counts[pair] = byte_pair_counts.get(pair, 0) + value
            bytes_to_word[pair].add(word)
    
    # 初始化vocab
    vocab = {i:bytes([i]) for i in range(256)}
    for j, special_token in enumerate(special_tokens):
        vocab[j+256] = special_token
    
    # merging process
    merges = []
    print(f"开始依次处理高频对，字典长度:{len(word_level_counts)}")
    
    for new_index in tqdm(range(len(vocab), vocab_size), desc="Merging Process"):
        # 获得top-1的邻接对排序，如果存在多个，则按字典序降序排序
        merge_tokens = max(byte_pair_counts.items(),key=lambda kv: (kv[1], vocab[kv[0][0]], vocab[kv[0][1]]))[0]
        
        # 当前要处理的token
        merge_token_bytes = vocab[merge_tokens[0]] + vocab[merge_tokens[1]]
        vocab[new_index] = merge_token_bytes
        pair = (vocab[merge_tokens[0]], vocab[merge_tokens[1]])
        merges.append(pair)
        
        # print(f"Merge {new_index}: {vocab[merge_tokens[0]]} + {vocab[merge_tokens[1]]} -> {merge_token_bytes}, freq: {byte_pair_counts[merge_tokens]}")

        # 遍历倒排索引
        for word in bytes_to_word[merge_tokens]:
            value = word_level_counts.get(word)
            byte_ls = word_to_bytes.get(word)
            if not byte_ls or len(byte_ls) < 2:
                continue
            new_byte_ls = []
            
            i = 0
            while i <= len(byte_ls) - 1:
                if i < len(byte_ls) - 1 and (byte_ls[i], byte_ls[i+1]) == (merge_tokens[0], merge_tokens[1]):
                    # 如果不是首位,且前一个字节不是new_index(避免重复扣减情况)
                    # case：
                        # [98, 114, 263, 103, 263, 103] -> [98, 114, 283, 283]
                        # (-): (114, 263), (103, 263)，如果不判断前一个字节是否为new_index，则(103, 263)会出现重复扣减的情况
                        # (+): (114, 283), (283, 283)
                    if i > 0 and new_byte_ls[-1] != new_index:
                        old_pair = (byte_ls[i-1], byte_ls[i])
                        byte_pair_counts[old_pair] -= value
                    
                    # 如果不是末位
                    if i < len(byte_ls) - 2:
                        old_pair = (byte_ls[i+1], byte_ls[i+2])
                        byte_pair_counts[old_pair] -= value
                    
                    new_byte_ls.append(new_index)
                    i += 2

                else:
                    new_byte_ls.append(byte_ls[i])
                    i += 1
            
            # 更新该word的字节表示
            word_to_bytes[word] = new_byte_ls 

            # 遍历新字节对，进行字典更新
            i = 0
            while i <= len(new_byte_ls) - 2:  
                if new_byte_ls[i] == new_index or new_byte_ls[i+1] == new_index:
                    pair = (new_byte_ls[i], new_byte_ls[i+1])
                    byte_pair_counts[pair] = byte_pair_counts.get(pair, 0) + value
                    bytes_to_word[pair].add(word)
                i += 1

        # 去掉该字节对
        del byte_pair_counts[merge_tokens]           
        
    return vocab, merges


if __name__ == '__main__':
    input_path = "../data/owt_train.txt"
    # input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 保存vocab 为json到本地
    vocab_str = {k: v.decode('utf-8', errors='ignore') for k, v in vocab.items()}
    
    # 保存 vocab 为 JSON 文件
    with open("./bpe_outputs/2.owt_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=4)
    
    with open("./bpe_outputs/2.owt_merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            token1 = merge[0].decode('utf-8', errors='ignore')
            token2 = merge[1].decode('utf-8', errors='ignore')
            f.write(f"{token1}\t{token2}\n")
    