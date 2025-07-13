from dataclasses import dataclass
from typing import Optional

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    cache_group_id: Optional[str] = None
