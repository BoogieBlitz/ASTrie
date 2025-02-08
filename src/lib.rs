use std::convert::TryInto;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockWriteGuard};

// Configuration constants
const TRIE_DEPTH_THRESHOLD: usize = 8; // Max trie depth before converting to B+ tree
const BTREE_MIN_OCCUPANCY: f32 = 0.4; // Minimum occupancy before collapsing to trie
const NODE_SIZE: usize = 256; // Size aligned with common CPU cache lines
const BTREE_MAX_KEYS: usize = NODE_SIZE / 2; // Maximum number of keys in a B+ tree node

// Trait for converting keys to bytes
pub trait ToBytes {
    fn to_bytes(&self) -> Vec<u8>;
}

// Implement ToBytes for common types
impl ToBytes for String {
    fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

impl ToBytes for str {
    fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

impl ToBytes for [u8] {
    fn to_bytes(&self) -> Vec<u8> {
        self.to_vec()
    }
}

impl ToBytes for Vec<u8> {
    fn to_bytes(&self) -> Vec<u8> {
        self.clone()
    }
}

// Implement for integer types
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

impl_to_bytes_for_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

// Main enum to represent either a trie or B+ tree node
enum NodeType<K: Clone + Ord, V> {
    Trie(TrieNode<K, V>),
    BTree(BTreeNode<K, V>),
}

// Trie node implementation
struct TrieNode<K: Clone + Ord, V> {
    children: Vec<Option<Arc<RwLock<NodeType<K, V>>>>>,
    value: Option<V>,
    depth: usize,
    size: AtomicUsize,
}

// B+ tree node implementation
struct BTreeNode<K: Clone + Ord, V> {
    keys: Vec<K>,
    values: Vec<V>,
    children: Vec<Arc<RwLock<NodeType<K, V>>>>,
    is_leaf: bool,
}

// Main ASTrie structure
pub struct ASTrie<K: Clone + Ord, V> {
    root: Arc<RwLock<NodeType<K, V>>>,
    size: AtomicUsize,
}

impl<K: Clone + Ord + ToBytes, V: Clone> ASTrie<K, V> {
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

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let mut current: Arc<RwLock<NodeType<K, V>>> = self.root.clone();
        let mut old_value: Option<V> = None;

        loop {
            // We'll handle the node type first, then decide what to do next
            let next_node: Arc<RwLock<NodeType<K, V>>> = {
                let mut node: RwLockWriteGuard<'_, NodeType<K, V>> = current.write().unwrap();

                match &mut *node {
                    NodeType::Trie(trie_node) => {
                        if trie_node.depth >= TRIE_DEPTH_THRESHOLD {
                            // Convert to B+ tree if depth threshold reached
                            let new_btree: NodeType<K, V> = self.convert_to_btree(trie_node);
                            *node = new_btree;
                            continue; // Retry insertion with new B+ tree node
                        }

                        // Get bytes from key for trie traversal
                        let key_bytes: Vec<u8> = self.key_to_bytes(&key);
                        let current_byte: u8 = key_bytes.get(trie_node.depth).copied().unwrap_or(0);

                        if trie_node.depth == key_bytes.len() {
                            // We've reached the end of the key, store the value
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

                        // Return the child node to traverse to
                        trie_node.children[current_byte as usize]
                            .as_ref()
                            .unwrap()
                            .clone()
                    }
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

                                    if btree_node.keys.len() > BTREE_MAX_KEYS {
                                        // Split node if it's too full
                                        self.split_btree_node(btree_node);
                                    } else if ((btree_node.keys.len() as f32) / (NODE_SIZE as f32))
                                        < BTREE_MIN_OCCUPANCY
                                    {
                                        // Convert back to trie if occupancy is too low
                                        *node = self.convert_to_trie(btree_node);
                                    }
                                    break;
                                }
                            }
                        } else {
                            // Handle internal node traversal
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

        self.size.fetch_add(1, Ordering::Relaxed);
        old_value
    }

    // Helper method to convert key to bytes for trie traversal
    fn key_to_bytes(&self, key: &K) -> Vec<u8> {
        key.to_bytes()
    }

    fn split_btree_node(&self, node: &mut BTreeNode<K, V>) {
        // Implementation remains the same...
        todo!("Implement B+ tree node splitting");
    }

    fn convert_to_btree(&self, trie_node: &TrieNode<K, V>) -> NodeType<K, V> {
        // Implementation for converting trie to B+ tree
        todo!("Implement trie to B+ tree conversion");
    }

    fn convert_to_trie(&self, btree_node: &BTreeNode<K, V>) -> NodeType<K, V> {
        todo!("Implement B+ tree to trie conversion");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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