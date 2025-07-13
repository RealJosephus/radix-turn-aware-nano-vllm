## Radix Tree KV Cache with Turn-Aware Growth

This repo extends [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by introducing a minimal implementation of a radix tree-based KV cache inspired by [SGLang](https://github.com/sgl-project/sglang), with enhancements tailored for dynamic chat scenarios.

### Features

* **Radix Tree Cache**: Efficient prefix-based reuse of KV cache entries.
* **Turn-Aware Growth**: Optimized for growing chat conversations with "turn"-based segmentation.
* **Zero-Copy Mid-Conversation Reuse**: Supports reuse of intermediate dialogue rounds even under non-prefix extension (e.g., duplex or interleaved KV layouts).
* **Minimal Overhead**: Designed for extensibility while maintaining lightweight performance.

### Use Case

Scenarios where conversation history may be updated or extended from arbitrary midpoints, especially in duplex or partially interleaved turn-based layouts.

You may also need an efficient attention kernel that supports dynamic RoPE or KV layouts for query computation. This repo only provides a minimal implementation of the KVCache.
