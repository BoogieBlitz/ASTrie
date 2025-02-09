//! # ASTrie (Adaptive Segmented Trie)
//!
//! `astrie` is a high-performance hybrid data structure that combines the benefits of tries and B+ trees
//! to provide efficient key-value storage with adaptive behavior based on data patterns.
//!
//! ## Features
//!
//! - **Hybrid Structure**: Dynamically switches between trie and B+ tree nodes based on data characteristics
//! - **Efficient Operations**: O(k) or O(log n) lookups, where k is key length
//! - **Range Queries**: Fast range scans with ordered iteration
//! - **Thread-Safe**: Concurrent access support using fine-grained locking
//! - **Memory Efficient**: Adaptive storage strategy to minimize memory usage
//! - **Generic**: Supports any key type that implements required traits
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! astrie = "0.1.0"
//! ```
//!
//! ## Example
//!
//! ```rust
//! use astrie::ASTrie;
//!
//! // Create a new ASTrie with string keys and integer values
//! let trie = ASTrie::<String, i32>::new();
//!
//! // Insert some data
//! trie.insert("hello".to_string(), 1);
//! trie.insert("world".to_string(), 2);
//!
//! // Lookup
//! assert_eq!(trie.get(&"hello".to_string()), Some(1));
//!
//! // Range query
//! let range = trie.range(&"h".to_string(), &"w".to_string());
//! assert_eq!(range.len(), 1);
//! ```
//!
//! ## Custom Types
//!
//! To use custom types as keys, implement the required traits:
//!
//! ```rust
//! use astrie::{ToBytes, FromBytes};
//!
//! #[derive(Clone, Ord, PartialOrd, Eq, PartialEq)]
//! struct CustomKey {
//!     id: u32,
//!     name: String,
//! }
//!
//! impl ToBytes for CustomKey {
//!     fn to_bytes(&self) -> Vec<u8> {
//!         let mut bytes = Vec::new();
//!         bytes.extend_from_slice(&self.id.to_be_bytes());
//!         bytes.extend_from_slice(self.name.as_bytes());
//!         bytes
//!     }
//! }
//!
//! impl FromBytes for CustomKey {
//!     fn from_bytes(bytes: &[u8]) -> Option<Self> {
//!         if bytes.len() < 4 {
//!             return None;
//!         }
//!         let id = u32::from_be_bytes(bytes[0..4].try_into().ok()?);
//!         let name = String::from_utf8(bytes[4..].to_vec()).ok()?;
//!         Some(CustomKey { id, name })
//!     }
//! }
//! ```
//!
//! ## Performance
//!
//! The ASTrie structure adapts to your data patterns:
//!
//! - For sparse key spaces: Uses trie nodes for O(k) lookups
//! - For dense key ranges: Uses B+ tree nodes for O(log n) lookups
//! - For range queries: O(log n + m) where m is the size of the range
//!
//! ## Thread Safety
//!
//! ASTrie uses fine-grained locking for concurrent access:
//!
//! ```rust
//! use std::thread;
//! use std::sync::Arc;
//! use astrie::ASTrie;
//!
//! let trie = Arc::new(ASTrie::<String, i32>::new());
//! let mut handles = vec![];
//!
//! for i in 0..10 {
//!     let trie_clone = trie.clone();
//!     let handle = thread::spawn(move || {
//!         trie_clone.insert(format!("key-{}", i), i);
//!     });
//!     handles.push(handle);
//! }
//!
//! for handle in handles {
//!     handle.join().unwrap();
//! }
//! ```
//!
//! ## Configuration
//!
//! Key constants that can be tuned:
//!
//! ```rust
//! const TRIE_DEPTH_THRESHOLD: usize = 8;    // Max trie depth before conversion
//! const BTREE_MIN_OCCUPANCY: f32 = 0.4;     // Min B+ tree node occupancy
//! const NODE_SIZE: usize = 256;             // Node size for cache alignment
//! ```
//!
//! ## Use Cases
//!
//! ASTrie is particularly well-suited for:
//!
//! - Key-value stores with range query requirements
//! - Network routing tables (IP prefix matching)
//! - Auto-complete systems
//! - Time-series databases
//! - In-memory caches
//!
//! ## Error Handling
//!
//! Operations that might fail return `Option` or `Result`:
//!
//! ```rust
//! use astrie::ASTrie;
//! let trie = ASTrie::<String, i32>::new();
//!
//! // Get returns Option
//! match trie.get(&"key".to_string()) {
//!     Some(value) => println!("Found: {}", value),
//!     None => println!("Key not found"),
//! }
//!
//! // Range queries return empty Vec if no matches
//! let empty_range = trie.range(&"z".to_string(), &"zzz".to_string());
//! assert!(empty_range.is_empty());
//! ```
//!
//! ## Implementation Details
//!
//! The adaptive behavior is controlled by two main mechanisms:
//!
//! 1. **Depth-based Conversion**: Trie nodes beyond `TRIE_DEPTH_THRESHOLD` are converted to B+ tree nodes
//! 2. **Occupancy-based Conversion**: B+ tree nodes below `BTREE_MIN_OCCUPANCY` are converted to tries
//!
//! ## License
//!
//! This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
//!
//! ## Contributing
//!
//! Contributions are welcome! Please feel free to submit a Pull Request.

mod utils;

use std::convert::TryInto;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

// Configuration constants

/// Max trie depth before converting to B+ tree
const TRIE_DEPTH_THRESHOLD: usize = 8;

/// Minimum occupancy before collapsing to trie
const BTREE_MIN_OCCUPANCY: f32 = 0.4;

/// Size aligned with common CPU cache lines
const NODE_SIZE: usize = 256;

/// Maximum number of keys in a B+ tree node
const BTREE_MAX_KEYS: usize = NODE_SIZE / 2;

/// Key trait for converting types to byte representation
pub trait ToBytes {
    fn to_bytes(&self) -> Vec<u8>;
}

// Implement ToBytes for String type
impl ToBytes for String {
    fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

// Implement ToBytes for string slice
impl ToBytes for str {
    fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

// Implement ToBytes for sequence of integers
impl ToBytes for [u8] {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_vec()
    }
}

// Implement ToBytes for Vector of integers
impl ToBytes for Vec<u8> {
    fn to_bytes(&self) -> Vec<u8> {
        self.clone()
    }
}

/// Implements ToBytes for integer types.
/// 
/// This macro automatically implements the ToBytes trait for the specified integer types.
/// The implementation converts the integer to its big-endian byte representation.
macro_rules! impl_to_bytes_for_int {
    ($($t:ty),*) => {
        $(
            impl ToBytes for $t {
                fn to_bytes(&self) -> Vec<u8> {
                    self.to_be_bytes().to_vec()
                }
            }
        )*
    }
}

// Macro implementation for converting integer types to byte representation
impl_to_bytes_for_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

/// Key trait for reconstructing types from bytes
pub trait FromBytes: Sized {
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
}

// Implement FromBytes for String type
impl FromBytes for String {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        String::from_utf8(bytes.to_vec()).ok()
    }
}

// Implement FromBytes for Vector of integers
impl FromBytes for Vec<u8> {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        Some(bytes.to_vec())
    }
}

/// Implements FromBytes for integer types.
/// 
/// This macro automatically implements the FromBytes trait for the specified integer types.
/// The implementation attempts to convert a slice of bytes back into the integer type,
/// expecting the bytes to be in big-endian order.
macro_rules! impl_from_bytes_for_int {
    ($($t:ty),*) => {
        $(
            impl FromBytes for $t {
                fn from_bytes(bytes: &[u8]) -> Option<Self> {
                    if bytes.len() == std::mem::size_of::<$t>() {
                        let array = bytes.try_into().ok()?;
                        Some(Self::from_be_bytes(array))
                    } else {
                        None
                    }
                }
            }
        )*
    }
}

// Macro implementation for reconstructing integer types from bytes
impl_from_bytes_for_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

/// Main enum to represent either a trie or B+ tree node
enum NodeType<K: Clone + Ord, V> {
    Trie(TrieNode<K, V>),
    BTree(BTreeNode<K, V>),
}

/// Contains information about a B+ tree node split: the median key-value pair and the new right node
struct SplitInfo<K: Clone + Ord + ToBytes, V: Clone> {
    median_key: K,
    median_value: V,
    right_node: Arc<RwLock<NodeType<K, V>>>,
}

/// Trie node implementation
struct TrieNode<K: Clone + Ord, V> {
    children: Vec<Option<Arc<RwLock<NodeType<K, V>>>>>,
    value: Option<V>,
    depth: usize,
    size: AtomicUsize,
}

/// B+ tree node implementation
struct BTreeNode<K: Clone + Ord, V> {
    keys: Vec<K>,
    values: Vec<V>,
    children: Vec<Arc<RwLock<NodeType<K, V>>>>,
    is_leaf: bool,
}

/// Main ASTrie data structure
pub struct ASTrie<K: Clone + Ord, V> {
    root: Arc<RwLock<NodeType<K, V>>>,
    size: AtomicUsize,
}

impl<K: Clone + Ord + ToBytes + FromBytes, V: Clone> ASTrie<K, V> {
    /// Creates a new empty ASTrie with default configuration
    pub fn new() -> Self {
        ASTrie {
            root: Arc::new(RwLock::new(NodeType::Trie(TrieNode {
                children: vec![None; 256], // Initialize with 256 possible branches
                value: None,
                depth: 0,
                size: AtomicUsize::new(0),
            }))),
            size: AtomicUsize::new(0),
        }
    }

    /// Retrieves the value associated with the given key, or None if the key doesn't exist
    pub fn get(&self, key: &K) -> Option<V> {
        // Convert the key to its byte representation for trie traversal
        let key_bytes: Vec<u8> = key.to_bytes();

        // Start at the root node, wrapped in Arc for thread-safe reference counting
        let mut current: Arc<RwLock<NodeType<K, V>>> = self.root.clone();

        // Continue traversing nodes until we find the key or determine it doesn't exist
        loop {
            // Create a new scope for the read lock to ensure it's released after node processing
            let next_node: Option<Arc<RwLock<NodeType<K, V>>>> = {
                // Acquire a read lock on the current node
                let node: RwLockReadGuard<'_, NodeType<K, V>> = current.read().unwrap();

                match &*node {
                    // If we're at a trie node
                    NodeType::Trie(trie_node) => {
                        // Check if we've consumed all bytes of the key
                        // If so, return the value at this node (if any)
                        if trie_node.depth == key_bytes.len() {
                            return trie_node.value.clone();
                        }

                        // Get the next byte from the key to determine which child to traverse
                        let current_byte: usize = key_bytes[trie_node.depth] as usize;

                        // Try to get the child node at the current byte's index
                        // If child exists, prepare to traverse to it
                        // If no child exists, the key doesn't exist in the trie
                        match &trie_node.children[current_byte] {
                            Some(child) => Some(child.clone()),
                            None => None,
                        }
                    }

                    // If we're at a B+ tree node
                    NodeType::BTree(btree_node) => {
                        // If this is a leaf node
                        if btree_node.is_leaf {
                            // Binary search in leaf node
                            match btree_node.keys.binary_search(key) {
                                Ok(idx) => return Some(btree_node.values[idx].clone()),
                                Err(_) => return None,
                            }
                        } else {
                            // Navigate internal node
                            // If key exists, go to the right child
                            // If key doesn't exist, go to the child where it would be
                            let child_idx: usize = match btree_node.keys.binary_search(key) {
                                Ok(idx) => idx + 1,
                                Err(idx) => idx,
                            };

                            // Return the child node to traverse
                            Some(btree_node.children[child_idx].clone())
                        }
                    }
                }
            }; // read lock is dropped here

            // Process the result of node traversal
            // If we have a next node to traverse, update current and continue
            // If we hit a dead end, the key doesn't exist
            match next_node {
                Some(next) => current = next,
                None => return None,
            }
        }
    }

    /// Returns all key-value pairs where the key is within the given range [start, end], inclusive
    pub fn range(&self, start: &K, end: &K) -> Vec<(K, V)> {
        // Initialize empty vector to store range query results
        let mut result: Vec<(K, V)> = Vec::new();

        // Early return if range is invalid (start > end)
        if start > end {
            return result;
        }

        // Start traversal from root
        let root: RwLockReadGuard<'_, NodeType<K, V>> = self.root.read().unwrap();
        match &*root {
            // If root is trie node, use trie range collection
            NodeType::Trie(trie_node) => {
                utils::collect_trie_range(trie_node, Vec::new(), start, end, &mut result);
            }

            // If root is B+ tree node, use B+ tree range collection
            NodeType::BTree(btree_node) => {
                utils::collect_btree_range(btree_node, start, end, &mut result);
            }
        }

        // Sort results to ensure correct order
        // (necessary because trie traversal might collect out of order)
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Inserts a key-value pair into the ASTrie, returning the previous value if it existed
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let mut current: Arc<RwLock<NodeType<K, V>>> = self.root.clone();
        let mut old_value: Option<V> = None;
        let mut path: Vec<(Arc<RwLock<NodeType<K, V>>>, usize)> = Vec::new();

        // Main insertion loop - continues until we either insert or hit an error
        loop {
            // Create a new scope for the write lock
            let next_node: Arc<RwLock<NodeType<K, V>>> = {
                // Acquire write lock on current node
                let mut node: RwLockWriteGuard<'_, NodeType<K, V>> = current.write().unwrap();

                // Match on the type of node (Trie or BTree)
                match &mut *node {
                    // Handle Trie node case
                    NodeType::Trie(trie_node) => {
                        // Check if we need to convert to B+ tree due to depth
                        if trie_node.depth >= TRIE_DEPTH_THRESHOLD {
                            // Convert to B+ tree if depth threshold reached
                            let new_btree: NodeType<K, V> = utils::convert_to_btree(trie_node);
                            *node = new_btree;
                            continue; // Retry insertion with new B+ tree node
                        }

                        // Get bytes from key for trie traversal
                        let key_bytes: Vec<u8> = utils::key_to_bytes(&key);
                        let current_byte: u8 = key_bytes.get(trie_node.depth).copied().unwrap_or(0);

                        // Check if we've reached the end of the key
                        if trie_node.depth == key_bytes.len() {
                            // Store value and return old value if it existed
                            old_value = trie_node.value.replace(value);
                            break;
                        }

                        // Create or traverse child node
                        if trie_node.children[current_byte as usize].is_none() {
                            trie_node.children[current_byte as usize] =
                                Some(Arc::new(RwLock::new(NodeType::Trie(TrieNode {
                                    children: vec![None; 256],
                                    value: None,
                                    depth: trie_node.depth + 1,
                                    size: AtomicUsize::new(0),
                                }))));
                        }

                        // Return child node for next iteration
                        trie_node.children[current_byte as usize]
                            .as_ref()
                            .unwrap()
                            .clone()
                    }

                    // Handle B+ tree node case
                    NodeType::BTree(btree_node) => {
                        if btree_node.is_leaf {
                            // Handle leaf node insertion
                            match btree_node.keys.binary_search(&key) {
                                Ok(idx) => {
                                    // Key exists, update value
                                    old_value =
                                        Some(std::mem::replace(&mut btree_node.values[idx], value));
                                    break;
                                }
                                Err(idx) => {
                                    // New key, insert at correct position
                                    btree_node.keys.insert(idx, key.clone());
                                    btree_node.values.insert(idx, value.clone());

                                    // Check if node needs splitting
                                    if btree_node.keys.len() > BTREE_MAX_KEYS {
                                        let split_info: Option<SplitInfo<K, V>> =
                                            utils::split_btree_node(btree_node);
                                        if let Some(split_info) = split_info {
                                            // Handle root split
                                            if path.is_empty() {
                                                let new_root: BTreeNode<K, V> = BTreeNode {
                                                    keys: vec![split_info.median_key],
                                                    values: vec![split_info.median_value],
                                                    children: vec![
                                                        current.clone(),
                                                        split_info.right_node,
                                                    ],
                                                    is_leaf: false,
                                                };
                                                *node = NodeType::BTree(new_root);
                                            } else {
                                                // Handle non-root split by updating parent
                                                let (parent, child_idx) = path.pop().unwrap();
                                                let mut parent: RwLockWriteGuard<'_, NodeType<K, V>> = parent.write().unwrap();
                                                if let NodeType::BTree(parent_node) = &mut *parent {
                                                    utils::handle_split(
                                                        parent_node,
                                                        child_idx,
                                                        split_info,
                                                    );
                                                }
                                            }
                                        }
                                    } else if ((btree_node.keys.len() as f32) / (NODE_SIZE as f32))
                                        < BTREE_MIN_OCCUPANCY
                                    {
                                        // Convert back to trie if occupancy is too low
                                        *node = utils::convert_to_trie(btree_node);
                                    }
                                    break;
                                }
                            }
                        } else {
                            // Handle internal node traversal to find appropriate child
                            let child_idx: usize = match btree_node.keys.binary_search(&key) {
                                Ok(idx) => idx + 1,
                                Err(idx) => idx,
                            };
                            btree_node.children[child_idx].clone()
                        }
                    }
                }
            }; // write lock is dropped here

            // Update current outside the write lock scope
            current = next_node;
        }

        // Update total size and return old value
        self.size.fetch_add(1, Ordering::Relaxed);
        old_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Utility function to create a test trie node
    fn create_test_trie_node<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
        value: Option<V>,
        depth: usize
    ) -> TrieNode<K, V> {
        TrieNode {
            children: vec![None; 256],
            value,
            depth,
            size: AtomicUsize::new(0),
        }
    }

    // Utility function to create a test B+ tree node
    fn create_test_btree_node<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
        keys: Vec<K>,
        values: Vec<V>,
        is_leaf: bool
    ) -> BTreeNode<K, V> {
        BTreeNode {
            keys,
            values,
            children: Vec::new(),
            is_leaf,
        }
    }

    #[test]
    fn test_collect_trie_range() {
        let mut result: Vec<(String, i32)> = Vec::new();
        let mut node: TrieNode<String, i32> = create_test_trie_node::<String, i32>(Some(1), 0);
        
        // Create a simple trie structure
        let child: TrieNode<String, i32> = create_test_trie_node(Some(2), 1);
        node.children[97] = Some(Arc::new(RwLock::new(NodeType::Trie(child)))); // 'a'

        utils::collect_trie_range(
            &node,
            Vec::new(),
            &"a".to_string(),
            &"c".to_string(),
            &mut result
        );

        assert_eq!(result.len(), 1);
        assert!(result.iter().any(|(k, v)| k == "a" && *v == 2));
    }

    #[test]
    fn test_collect_btree_range() {
        let mut result: Vec<(i32, String)> = Vec::new();
        let node: BTreeNode<i32, String> = create_test_btree_node(
            vec![1, 3, 5],
            vec!["one".to_string(), "three".to_string(), "five".to_string()],
            true
        );

        utils::collect_btree_range(&node, &1, &4, &mut result);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, "one".to_string()));
        assert_eq!(result[1], (3, "three".to_string()));
    }

    #[test]
    fn test_insert_into_trie() {
        let mut root: TrieNode<String, i32> = create_test_trie_node::<String, i32>(None, 0);
        let key: String = "test".to_string();
        let value: i32 = 42;
        let key_bytes: &[u8] = key.as_bytes();

        utils::insert_into_trie(&mut root, &key, value, 0, key_bytes);

        // Verify the insertion
        fn verify_path(
            node: &Option<Arc<RwLock<NodeType<String, i32>>>>,
            remaining_bytes: &[u8],
            expected_value: i32
        ) -> bool {
            match node {
                None => false,
                Some(child) => {
                    if let Ok(guard) = child.read() {
                        match &*guard {
                            NodeType::Trie(trie_node) => {
                                if remaining_bytes.is_empty() {
                                    return trie_node.value == Some(expected_value);
                                }
                                let next_byte = remaining_bytes[0] as usize;
                                verify_path(
                                    &trie_node.children[next_byte],
                                    &remaining_bytes[1..],
                                    expected_value
                                )
                            },
                            _ => false,
                        }
                    } else {
                        false
                    }
                }
            }
        }

        // Start verification from root's children
        let first_byte = key_bytes[0] as usize;
        assert!(verify_path(&root.children[first_byte], &key_bytes[1..], value));
    }

    #[test]
    fn test_collect_leaf_pairs() {
        let mut pairs: Vec<(i32, String)> = Vec::new();
        let leaf1: BTreeNode<i32, String> = create_test_btree_node(
            vec![1, 2],
            vec!["one".to_string(), "two".to_string()],
            true
        );
        let leaf2: BTreeNode<i32, String> = create_test_btree_node(
            vec![3, 4],
            vec!["three".to_string(), "four".to_string()],
            true
        );

        let mut parent: BTreeNode<i32, String> = create_test_btree_node(vec![3], vec!["three".to_string()], false);
        parent.children = vec![
            Arc::new(RwLock::new(NodeType::BTree(leaf1))),
            Arc::new(RwLock::new(NodeType::BTree(leaf2))),
        ];

        utils::collect_leaf_pairs(&parent, &mut pairs);

        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], (1, "one".to_string()));
        assert_eq!(pairs[3], (4, "four".to_string()));
    }

    #[test]
    fn test_handle_split() {
        // Create a parent node with initial children
        let mut parent = BTreeNode {
            keys: vec![1, 5],
            values: vec!["one".to_string(), "five".to_string()],
            children: vec![
                Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
                    keys: vec![0],
                    values: vec!["zero".to_string()],
                    children: Vec::new(),
                    is_leaf: true,
                }))),
                Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
                    keys: vec![2],
                    values: vec!["two".to_string()],
                    children: Vec::new(),
                    is_leaf: true,
                }))),
                Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
                    keys: vec![6],
                    values: vec!["six".to_string()],
                    children: Vec::new(),
                    is_leaf: true,
                }))),
            ],
            is_leaf: false,
        };

        // Create the right node that results from a split
        let right_node = Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
            keys: vec![4],
            values: vec!["four".to_string()],
            children: Vec::new(),
            is_leaf: true,
        })));

        // Create split info
        let split_info = SplitInfo {
            median_key: 3,
            median_value: "three".to_string(),
            right_node: right_node.clone(),
        };

        // Handle the split at index 1 (between 1 and 5)
        utils::handle_split(&mut parent, 1, split_info);

        // Verify the result
        assert_eq!(parent.keys, vec![1, 3, 5], "Keys should be [1, 3, 5]");
        assert_eq!(
            parent.values,
            vec!["one".to_string(), "three".to_string(), "five".to_string()],
            "Values should be [one, three, five]"
        );
        assert_eq!(parent.children.len(), 4, "Should have 4 children after split");

        // Helper function to safely verify a child node
        fn verify_child_node(child: &Arc<RwLock<NodeType<i32, String>>>, expected_keys: Vec<i32>, expected_values: Vec<String>) -> bool {
            if let Ok(guard) = child.read() {
                match &*guard {
                    NodeType::BTree(node) => {
                        node.keys == expected_keys && node.values == expected_values
                    },
                    _ => false
                }
            } else {
                false
            }
        }

        // Verify children order
        assert!(verify_child_node(&parent.children[0], vec![0], vec!["zero".to_string()]), 
            "First child should have key [0]");
        assert!(verify_child_node(&parent.children[1], vec![2], vec!["two".to_string()]), 
            "Second child should have key [2]");
        assert!(verify_child_node(&parent.children[2], vec![4], vec!["four".to_string()]), 
            "Third child should have key [4]");
        assert!(verify_child_node(&parent.children[3], vec![6], vec!["six".to_string()]), 
            "Fourth child should have key [6]");
    }

    #[test]
    fn test_split_btree_node() {
        // Test leaf node split
        let mut leaf: BTreeNode<i32, String> = BTreeNode {
            keys: vec![1, 2, 3, 4, 5],
            values: vec![
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
                "4".to_string(),
                "5".to_string()
            ],
            children: Vec::new(),
            is_leaf: true,
        };

        let split_info: Option<SplitInfo<i32, String>> = utils::split_btree_node(&mut leaf);
        assert!(split_info.is_some());
        let info: SplitInfo<i32, String> = split_info.unwrap();

        // For a leaf node:
        // - Left node (original) should have [1, 2]
        // - Right node should have [3, 4, 5]
        // - Median key should be 3 (first key of right node)
        assert_eq!(leaf.keys, vec![1, 2]);
        assert_eq!(leaf.values, vec!["1".to_string(), "2".to_string()]);

        // Verify the right node
        if let Ok(guard) = info.right_node.read() {
            match &*guard {
                NodeType::BTree(right) => {
                    assert_eq!(right.keys, vec![3, 4, 5]);
                    assert_eq!(right.values, vec![
                        "3".to_string(),
                        "4".to_string(),
                        "5".to_string()
                    ]);
                    assert!(right.is_leaf);
                },
                _ => panic!("Expected BTree node"),
            }
        }
        assert_eq!(info.median_key, 3);
        assert_eq!(info.median_value, "3".to_string());

        // Test internal node split
        let mut internal: BTreeNode<i32, String> = BTreeNode {
            keys: vec![10, 20, 30, 40],
            values: vec![
                "10".to_string(),
                "20".to_string(),
                "30".to_string(),
                "40".to_string()
            ],
            children: vec![
                Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
                    keys: vec![5],
                    values: vec!["5".to_string()],
                    children: Vec::new(),
                    is_leaf: true,
                }))),
                Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
                    keys: vec![15],
                    values: vec!["15".to_string()],
                    children: Vec::new(),
                    is_leaf: true,
                }))),
                Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
                    keys: vec![25],
                    values: vec!["25".to_string()],
                    children: Vec::new(),
                    is_leaf: true,
                }))),
                Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
                    keys: vec![35],
                    values: vec!["35".to_string()],
                    children: Vec::new(),
                    is_leaf: true,
                }))),
                Arc::new(RwLock::new(NodeType::BTree(BTreeNode {
                    keys: vec![45],
                    values: vec!["45".to_string()],
                    children: Vec::new(),
                    is_leaf: true,
                }))),
            ],
            is_leaf: false,
        };

        let split_info: Option<SplitInfo<i32, String>> = utils::split_btree_node(&mut internal);
        assert!(split_info.is_some());
        let info: SplitInfo<i32, String> = split_info.unwrap();

        // For an internal node:
        // - Left node should have [10, 20]
        // - Right node should have [30, 40]
        // - Median key (20) moves up
        assert_eq!(internal.keys, vec![10, 20]);
        assert_eq!(internal.values, vec!["10".to_string(), "20".to_string()]);
        assert_eq!(internal.children.len(), 3);

        if let Ok(guard) = info.right_node.read() {
            match &*guard {
                NodeType::BTree(right) => {
                    assert_eq!(right.keys, vec![40]);
                    assert_eq!(right.values, vec!["40".to_string()]);
                    assert_eq!(right.children.len(), 2);
                    assert!(!right.is_leaf);
                },
                _ => panic!("Expected BTree node"),
            }
        }
        assert_eq!(info.median_key, 30);
        assert_eq!(info.median_value, "30".to_string());
    }

    #[test]
    fn test_collect_pairs() {
        let mut pairs: Vec<(String, i32)> = Vec::new();
        let mut node: TrieNode<String, i32> = create_test_trie_node::<String, i32>(Some(1), 0);
        
        // Add some children
        let child1: TrieNode<String, i32> = create_test_trie_node(Some(2), 1);
        let child2: TrieNode<String, i32> = create_test_trie_node(Some(3), 1);
        
        node.children[97] = Some(Arc::new(RwLock::new(NodeType::Trie(child1)))); // 'a'
        node.children[98] = Some(Arc::new(RwLock::new(NodeType::Trie(child2)))); // 'b'

        utils::collect_pairs(&node, Vec::new(), &mut pairs);

        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn test_convert_to_trie() {
        let btree: BTreeNode<String, i32> = create_test_btree_node(
            vec!["a".to_string(), "b".to_string()],
            vec![1, 2],
            true
        );

        let result: NodeType<String, i32> = utils::convert_to_trie(&btree);
        
        match result {
            NodeType::Trie(trie) => {
                // Verify first key-value pair
                let key_bytes: &[u8] = "a".as_bytes();
                
                fn verify_trie_path(
                    children: &[Option<Arc<RwLock<NodeType<String, i32>>>>],
                    bytes: &[u8],
                    expected_value: i32
                ) -> bool {
                    if bytes.is_empty() {
                        return false;
                    }

                    let byte: usize = bytes[0] as usize;
                    if let Some(ref child) = children[byte] {
                        if let Ok(guard) = child.read() {
                            match &*guard {
                                NodeType::Trie(trie_node) => {
                                    if bytes.len() == 1 {
                                        trie_node.value == Some(expected_value)
                                    } else {
                                        verify_trie_path(&trie_node.children, &bytes[1..], expected_value)
                                    }
                                },
                                _ => false,
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }

                assert!(verify_trie_path(&trie.children, key_bytes, 1));
            },
            _ => panic!("Expected Trie node"),
        }
    }

    #[test]
    fn test_convert_to_btree() {
        // Explicitly specify type parameters
        let mut trie: TrieNode<String, i32> = create_test_trie_node(Some(1), 0);
        
        // Add some children
        let child: TrieNode<String, i32> = create_test_trie_node(Some(2), 1);
        trie.children[97] = Some(Arc::new(RwLock::new(NodeType::Trie(child))));

        let result: NodeType<String, i32> = utils::convert_to_btree(&trie);
        
        match result {
            NodeType::BTree(btree) => {
                assert!(btree.is_leaf);
                assert_eq!(btree.keys.len(), 2);
                assert_eq!(btree.values.len(), 2);
            },
            _ => panic!("Expected BTree node"),
        }
    }

    #[test]
    fn test_btree_node_operations() {
        // Test with explicit types for B+ tree nodes
        let btree: BTreeNode<String, i32> = create_test_btree_node(
            vec!["a".to_string(), "b".to_string()],
            vec![1, 2],
            true
        );

        assert_eq!(btree.keys.len(), 2);
        assert_eq!(btree.values.len(), 2);
        assert!(btree.is_leaf);
    }

    #[test]
    fn test_string_key_conversion() {
        let key = String::from("hello");
        assert_eq!(key.to_bytes(), b"hello");
    }

    #[test]
    fn test_integer_key_conversion() {
        let key: u32 = 42;
        assert_eq!(key.to_bytes(), vec![0, 0, 0, 42]);

        let key: i16 = -1;
        assert_eq!(key.to_bytes(), vec![255, 255]);
    }

    #[test]
    fn test_bytes_key_conversion() {
        let key = vec![1, 2, 3, 4];
        assert_eq!(key.to_bytes(), vec![1, 2, 3, 4]);
    }
}
