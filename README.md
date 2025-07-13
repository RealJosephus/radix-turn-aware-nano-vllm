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

An example of the complete system is shown in the diagram below. Note that this repo only implements the "Shared KV Cache" portion. The other diagrams are for reference only, illustrating a potential implementation path (not optimal) to provide a high-level overview of the entire system.

<img width="8192" height="5277" alt="diagram" src="https://github.com/user-attachments/assets/2f028d58-0bf0-4393-b8bc-1914031ed27f" />
