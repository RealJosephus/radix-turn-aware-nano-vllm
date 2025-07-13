from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, verbose: bool = False):
        self.verbose = verbose
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        temp_waiting = deque()

        while self.waiting:
            seq = self.waiting.popleft()
            if num_seqs >= self.max_num_seqs:
                temp_waiting.append(seq)
                continue

            if self.block_manager.match_and_allocate(seq):
                if len(seq) > 0 and seq.num_cached_tokens == len(seq):
                    if self.verbose:
                        print(
                            f"[Scheduler] Scheduling Seq {seq.seq_id} (Prefill): "
                            f"Total tokens: {len(seq)}, "
                            f"Cached tokens: {seq.num_cached_tokens} "
                            f"-> Caching Hit: True (100% - SKIPPING PREFILL)"
                        )
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                    continue

                uncached_tokens = len(seq) - seq.num_cached_tokens
                if num_batched_tokens + uncached_tokens > self.max_num_batched_tokens:
                    self.block_manager.deallocate(seq)
                    temp_waiting.append(seq)
                    continue

                if self.verbose:
                    print(
                        f"[Scheduler] Scheduling Seq {seq.seq_id} (Prefill): "
                        f"Total tokens: {len(seq)}, "
                        f"Cached tokens: {seq.num_cached_tokens} "
                        f"-> Caching Hit: {seq.num_cached_tokens > 0}"
                    )

                num_seqs += 1
                num_batched_tokens += uncached_tokens
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_seqs.append(seq)
            else:
                temp_waiting.append(seq)

        self.waiting.extendleft(temp_waiting)

        if scheduled_seqs:
            return scheduled_seqs, True

        if not self.running:
             return [], False

        scheduled_seqs.clear()
        num_seqs = 0
        running_list = list(self.running)
        self.running = deque([s for s in self.running if s not in running_list[:self.max_num_seqs]])

        for seq in running_list:
            if num_seqs >= self.max_num_seqs:
                self.running.appendleft(seq)
                continue

            try:
                self.block_manager.may_append(seq)
                num_seqs += 1
                scheduled_seqs.append(seq)
            except MemoryError:
                self.preempt(seq)
                if scheduled_seqs:
                    self.preempt(scheduled_seqs.pop())
                break

        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
