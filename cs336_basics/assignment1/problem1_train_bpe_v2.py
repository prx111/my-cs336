"""
v2修改: 
https://zhuanlan.zhihu.com/p/1920487178846344415
1, 使用优先队列可以动态维护出现次数最多的 pair, 实现高效查找。这在 pair 数较多的情况下非常有效。
2, 如果用 list 存储每个 word 及其包含的 tokens, 每次合并要把包含目标 pair 的全部 word 重构，复杂度和 word 包含的 token 数有关。
   如果改用双向链表，可以实现 O(1) 的原地更新。

本质是空间换时间，多了一个堆，维护字节对频率 以及 排序 过程 均在堆进行。

v2 bug fixed: 原代码使用集合维护倒排索引, 由于集合无序, 导致不会按顺序记录同一单词的node; 在遍历倒排索引时, 一旦遍历的索引集合中的节点被删除, 需要做判断。
badcase: 单词的token序列: (46, 46, 46, 46, 318). 
    1) 需要保证储存(46, 46)的倒排索引中, 在该单词内部是有序的, 否则不会正确合并。
    2) 在合并过程中, 可能会先合并掉第一个和第二个(46, 46), 导致第三个 46 node被删除, 但仍然被遍历到, 需要判断避免访问已经被删除的node.
"""

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
import heapq

class Node:
    def __init__(self, value, freq, prev=None, next_=None):
        self.value = value
        self.freq = freq
        self.prev = prev
        self.next_ = next_

class Item:
    def __init__(self, freq, id_pair, byte_pair):
        self.freq = freq
        self.id_pair = id_pair
        self.byte_pair = byte_pair
    
    def __lt__(self, other):
        if self.freq == other.freq:
            return self.byte_pair > other.byte_pair
        else:
            return self.freq > other.freq


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
        2.1 使用双向链表 重写 word_to_bytes.
        2.2 构造倒排索引 为 dict 只储存 某个pair 对应的node
        2.3 完成2.1和2.2后，使用字典记录 所有pair 的频率，记录完之后，将这些对放入大顶堆中
    
    3. 初始化 vocab，包含256个单字节和特殊token
    4. 开始合并，每次从 byte_pair_counts找到频率最高的字节对；
        4.1 从大顶堆中获取频率最高的字节对。因为大顶堆没有del，需要判断该对是否在倒排索引中

    5. 遍历所有单词，查看该单词是否存在当前匹配的字节对.
        如果存在，则进行合并，并更新该单词的字节表示，同时更新 byte_pair_counts 中相关字节对的频率
        扣除一部分字节对，更新一部分字节对。
        5.1 基于倒排索引遍历所有相干的单词
          5.2 扣减老字节对频率，更新堆；  增加新字节对频率，更新堆
    
    6. byte_pair_counts中去掉合并的字节对，继续下一轮合并，直到达到 vocab_size
        6.1 删除 倒排索引 和 字节对频率 对应的key
 

    返回：
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    
    """
    
    special_tokens = [token.encode('utf-8') for token in special_tokens]
    word_level_counts = process_file(input_path)
    
    
    word_to_bytes = {}
    bytes_to_node = defaultdict(dict)    # 倒排索引,使用dict确保有序

    # 初始化vocab
    vocab = {i:bytes([i]) for i in range(256)}
    for j, special_token in enumerate(special_tokens):
        vocab[j+256] = special_token

    byte_pair_counts = {}
    for word, word_freq in tqdm(word_level_counts.items(), total=len(word_level_counts), desc="分别统计字节对频率、词到字节映射、倒排索引"):   
        byte_ls = list(word.encode('utf-8'))
        if len(byte_ls) < 2:
            continue
        
        head = Node(byte_ls[0], word_freq)
        word_to_bytes[word] = head
        current = head  
        
        for i in range(1, len(byte_ls)):
            node = Node(byte_ls[i], word_freq, current)
            current.next_ = node

            id_pair = (current.value, node.value)
            byte_pair_counts[id_pair] =  byte_pair_counts.get(id_pair, 0) + word_freq
            
            bytes_to_node[id_pair][current] = True
            current = node

            

    # new: 初始化堆，默认是小顶堆，需要在比较函数处添加调整
    heap = []
    for pair, freq in byte_pair_counts.items():
        heapq.heappush(heap, Item(
            freq,
            pair,
            (vocab[pair[0]], vocab[pair[1]])
        ))
    
    # merging process
    merges = []
    print(f"开始依次处理高频对，字典长度:{len(word_level_counts)}")
    
    for new_index in tqdm(range(len(vocab), vocab_size), desc="Merging Process"):
        # 获得top-1的邻接对排序，如果存在多个，则按字典序降序排序
        if len(heap) <= 0:
            break
        while heap:
            # 因为堆只有【添加】，没有【调整】，所以需要确认弹出的对是最新的
            item = heapq.heappop(heap)
            if item.id_pair not in byte_pair_counts:
                continue
            if item.freq != byte_pair_counts[item.id_pair]:
                continue
            else:
                break
        
        # 当前要处理的token
        id_pair, byte_pair = item.id_pair, item.byte_pair
        merges.append(byte_pair)
        merge_token_bytes = byte_pair[0] + byte_pair[1]
        vocab[new_index] = merge_token_bytes

        # 遍历倒排索引 
        for node in list(bytes_to_node[id_pair].keys()):

            # 为了避免node已经被pop同时仍然被遍历到的情况。
            if node not in bytes_to_node[id_pair]:
                continue
            
            word_freq = node.freq
            right_node = node.next_
            if right_node is None:
                continue

            # 如果不是首位：
            if node.prev:
                # 扣减 左侧 老对
                prev_node = node.prev
                left_pair = (prev_node.value, node.value)
                byte_pair_counts[left_pair] -= word_freq
                left_byte_pair = (vocab[left_pair[0]], vocab[left_pair[1]])
                heapq.heappush(heap, Item(
                    byte_pair_counts[left_pair],
                    left_pair,
                    left_byte_pair
                ))

                # 处理新对
                bytes_to_node[left_pair].pop(prev_node)
                add_pair = (prev_node.value, new_index)
                bytes_to_node[add_pair][prev_node] = True
                byte_pair_counts[add_pair] = byte_pair_counts.get(add_pair, 0) + word_freq
                add_byte_pair = (vocab[add_pair[0]], vocab[add_pair[1]])
                heapq.heappush(heap, Item(
                    byte_pair_counts[add_pair],
                    add_pair,
                    add_byte_pair
                ))
            
            if right_node.next_:
                # 扣减 右侧 老对
                right_pair = (right_node.value, right_node.next_.value)
                byte_pair_counts[right_pair] -= word_freq
                right_byte_pair = (vocab[right_pair[0]], vocab[right_pair[1]])
                heapq.heappush(heap, Item(
                    byte_pair_counts[right_pair],
                    right_pair,
                    right_byte_pair
                ))

                # 处理新对
                add_pair = (new_index, right_node.next_.value)
                bytes_to_node[right_pair].pop(right_node)
                bytes_to_node[add_pair][node] = True
                byte_pair_counts[add_pair] = byte_pair_counts.get(add_pair, 0) + word_freq
                add_byte_pair = (vocab[add_pair[0]], vocab[add_pair[1]])
                heapq.heappush(heap, Item(
                    byte_pair_counts[add_pair],
                    add_pair,
                    add_byte_pair
                ))
            
            node.value = new_index
            node.next_ = right_node.next_
            if right_node.next_:
                right_node.next_.prev = node
            

        # 去掉该字节对
        del byte_pair_counts[id_pair]
        del bytes_to_node[id_pair]           
        
    return vocab, merges


if __name__ == '__main__':
    os.chdir('/root/autodl-tmp/assignment1-basics/cs336_basics')    # debug mode
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
    