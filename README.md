# ASTrie (Adaptive Segmented Trie)
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

## High-Level Algorithm

### Core Data Structure

**1. Hybrid Structure**

- Root starts as a trie node
- Nodes can be either trie nodes or B+ tree nodes
- Dynamic conversion between types based on usage patterns

**2. Node Types**

```rust
enum NodeType {
    Trie(TrieNode),    // For sparse/prefix-heavy data
    BTree(BTreeNode)   // For dense data ranges
}
```

### Core Operations

**1. Insertion**

```
insert(key, value):
1. Start at root node
2. For each node encountered:
   If Trie Node:
     - If depth > TRIE_DEPTH_THRESHOLD:
       → Convert to B+ tree (convert_to_btree)
     - Otherwise:
       → Navigate using key bytes
       → Create child nodes as needed
   
   If B+ Tree Node:
     - If leaf node:
       → Insert key-value pair
       → If too full: split node
       → If too sparse: convert to trie (convert_to_trie)
     - If internal node:
       → Navigate to correct child
```

**2. Lookup**

```
get(key):
1. Start at root node
2. For each node encountered:
   If Trie Node:
     → Use key bytes to navigate
     → Return value if found at leaf
   
   If B+ Tree Node:
     → Binary search for key
     → Navigate to child if internal node
     → Return value if found at leaf
```

**3. Range Query**

```
range(start_key, end_key):
1. Find path to start_key
2. For each node in path:
   If Trie Node:
     → Collect all values in range via DFS
   
   If B+ Tree Node:
     → Use B+ tree's natural ordering
     → Scan leaf nodes using links
```

### Adaptive Mechanisms

**1. Trie → B+ Tree Conversion**

```
convert_to_btree(trie_node):
1. Recursively collect all key-value pairs
2. Sort pairs by key
3. Create B+ tree leaf node
4. If too many keys:
   → Split into multiple nodes
   → Create parent nodes as needed
```

**2. B+ Tree → Trie Conversion**

```
convert_to_trie(btree_node):
1. Create empty trie root
2. For each key-value pair:
   → Convert key to byte path
   → Create trie path
   → Store value at leaf
3. Balance if needed
```

## Performance Characteristics

### Space Complexity

- Trie: O(k*n) where k is key length
- B+ Tree: O(n) where n is number of items
- Adaptive: O(min(k*n, n)) depending on structure

### Time Complexity

**1. Insertions**

- Best: O(1) for trie
- Worst: O(log n) for B+ tree
- Amortized: O(log n)

**2. Lookups**

- Best: O(k) for trie where k is key length
- Worst: O(log n) for B+ tree
- Average: O(min(k, log n))

**3. Range Queries**

- O(log n + m) where m is number of items in range
