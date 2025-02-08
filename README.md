# ASTrie
A hybrid data structure that efficiently supports fast lookups, insertions, and range queries with an adaptive mechanism to balance memory usage and performance.

## Key Features
**1. Trie & B+ Tree Hybrid**

- Uses trie-based prefix matching for fast key lookups.
- Segments the trie into B+ tree nodes when prefixes grow beyond a threshold, improving memory efficiency.

**2. Adaptive Segmentation**

- If a trie branch gets too deep, it automatically converts into a B+ tree to avoid memory fragmentation.
- If a B+ tree node becomes too sparse, it collapses back into a trie for optimized storage.

**3. Parallelization with Rust’s Ownership Model**

- Designed with lock-free concurrency using atomic operations (e.g., Arc<RwLock<T>>).
- Parallel insertions & lookups without performance degradation.

**4. Efficient Range Queries**

- Unlike traditional tries, the B+ tree nodes allow ordered key traversal, making range queries (e.g., key-value databases) much more efficient.

**5. Cache-aware Structure**

- Designed to maximize CPU cache locality, reducing cache misses.
- Prefetching mechanisms to speed up queries.

## Use Cases

- Key-Value Stores (faster and more memory-efficient alternative to HashMap)
- Network Routing Tables (efficient prefix-matching for IP lookups)
- Inverted Indexes for Search Engines
- Auto-complete systems (efficient prefix-based lookups)
- Time-series Databases (optimized range queries)
