from enum import Enum, auto
from itertools import count, chain
from typing import Optional, List

from nanovllm.sampling_params import SamplingParams
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nanovllm.engine.block_manager import CacheNode


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Turn:
    def __init__(self, token_ids: list[int], block_size: int):
        self.token_ids = token_ids
        self._block_size = block_size

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        if not self.token_ids:
            return 0
        return (self.num_tokens + self._block_size - 1) // self._block_size

    def __len__(self) -> int:
        return self.num_tokens


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams, im_start_id: int):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.cache_group_id = sampling_params.cache_group_id
        self.turns: list[Turn] = []
        if token_ids:
            current_turn_start_idx = 0
            for i, token_id in enumerate(token_ids):
                if i > 0 and token_id == im_start_id:
                    self.turns.append(Turn(token_ids[current_turn_start_idx:i], self.block_size))
                    current_turn_start_idx = i
            self.turns.append(Turn(token_ids[current_turn_start_idx:], self.block_size))

        self.num_prompt_tokens = len(token_ids)
        self.num_tokens = self.num_prompt_tokens
        self.block_table: list[int] = []
        self.turn_cache_nodes: List[Optional['CacheNode']] = [None] * len(self.turns)
        self.num_cached_tokens: int = 0
        self.completion_token_ids: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def token_ids(self) -> list[int]:
        prompt_tokens = list(chain.from_iterable(t.token_ids for t in self.turns))
        return prompt_tokens + self.completion_token_ids

    @property
    def last_token(self) -> int:
        if not self.token_ids: return -1
        return self.token_ids[-1]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return len(self.completion_token_ids)

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_tokens == 0:
            return 0
        if self.num_tokens % self.block_size == 0:
            return self.block_size
        return self.num_tokens % self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        start_idx = i * self.block_size
        end_idx = min(start_idx + self.block_size, self.num_tokens)
        return self.token_ids[start_idx:end_idx]

    def append_token(self, token_id: int):
        self.completion_token_ids.append(token_id)
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if not self.completion_token_ids else self.last_token)

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table, token_data) = state
        if isinstance(token_data, list):
             self._flat_token_ids_for_worker = token_data
        else:
             self._last_token_for_worker = token_data
