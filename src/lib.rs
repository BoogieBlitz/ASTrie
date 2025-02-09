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

// Trait for converting back from bytes
pub trait FromBytes: Sized {
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
}

// Implement FromBytes for common types
impl FromBytes for String {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        String::from_utf8(bytes.to_vec()).ok()
    }
}

impl FromBytes for Vec<u8> {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        Some(bytes.to_vec())
    }
}

// Implement FromBytes for integer types
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

impl_from_bytes_for_int!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

// Main enum to represent either a trie or B+ tree node
enum NodeType<K: Clone + Ord, V> {
    Trie(TrieNode<K, V>),
    BTree(BTreeNode<K, V>),
}

struct SplitInfo<K: Clone + Ord + ToBytes, V: Clone> {
    median_key: K,
    median_value: V,
    right_node: Arc<RwLock<NodeType<K, V>>>,
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

impl<K: Clone + Ord + ToBytes + FromBytes, V: Clone> ASTrie<K, V> {
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
        let mut path: Vec<(Arc<RwLock<NodeType<K, V>>>, usize)> = Vec::new();

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
                                        let split_info: Option<SplitInfo<K, V>> =
                                            self.split_btree_node(btree_node);
                                        if let Some(split_info) = split_info {
                                            // Handle root split
                                            if path.is_empty() {
                                                let new_root = BTreeNode {
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
                                                let mut parent = parent.write().unwrap();
                                                if let NodeType::BTree(parent_node) = &mut *parent {
                                                    self.handle_split(
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

    // Helper method to handle the split info and update parent node
    fn handle_split(
        &self,
        parent: &mut BTreeNode<K, V>,
        child_idx: usize,
        split_info: SplitInfo<K, V>,
    ) {
        // Insert the median key and value at the correct position
        parent.keys.insert(child_idx, split_info.median_key);
        parent.values.insert(child_idx, split_info.median_value);

        // Insert the new right child
        parent.children.insert(child_idx + 1, split_info.right_node);
    }

    fn split_btree_node(&self, node: &mut BTreeNode<K, V>) -> Option<SplitInfo<K, V>> {
        let mid: usize = BTREE_MAX_KEYS / 2;

        // Create a new right node
        let mut right_node: BTreeNode<K, V> = BTreeNode {
            keys: Vec::with_capacity(BTREE_MAX_KEYS),
            values: Vec::with_capacity(BTREE_MAX_KEYS),
            children: Vec::with_capacity(BTREE_MAX_KEYS + 1),
            is_leaf: node.is_leaf,
        };

        // For leaf nodes
        if node.is_leaf {
            // Move half of the keys and values to the right node
            right_node.keys = node.keys.split_off(mid);
            right_node.values = node.values.split_off(mid);

            // Create SplitInfo with the first key of right node (no key is moved up)
            let median_key: K = right_node.keys[0].clone();
            let median_value: V = right_node.values[0].clone();

            // Wrap the right node
            let right_node = Arc::new(RwLock::new(NodeType::BTree(right_node)));

            Some(SplitInfo {
                median_key,
                median_value,
                right_node,
            })
        }
        // For internal nodes
        else {
            // Save the median key and value (they will move up)
            let median_key: K = node.keys[mid].clone();
            let median_value: V = node.values[mid].clone();

            // Move keys and values after median to right node
            right_node.keys = node.keys.split_off(mid + 1);
            right_node.values = node.values.split_off(mid + 1);

            // Remove the median key and value from the left node
            node.keys.pop();
            node.values.pop();

            // Move the corresponding children
            right_node.children = node.children.split_off(mid + 1);

            // Wrap the right node
            let right_node: Arc<RwLock<NodeType<K, V>>> =
                Arc::new(RwLock::new(NodeType::BTree(right_node)));

            Some(SplitInfo {
                median_key,
                median_value,
                right_node,
            })
        }
    }

    // Helper method to collect key-value pairs from trie
    fn collect_pairs(&self, node: &TrieNode<K, V>, prefix: Vec<u8>, pairs: &mut Vec<(K, V)>) {
        // If this node has a value, reconstruct the key and add the pair
        if let Some(value) = &node.value {
            if let Some(key) = K::from_bytes(&prefix) {
                pairs.push((key, value.clone()));
            }
        }

        // Recursively traverse all children
        for (byte, child_opt) in node.children.iter().enumerate() {
            if let Some(child) = child_opt {
                let mut new_prefix = prefix.clone();
                new_prefix.push(byte as u8);

                if let Ok(guard) = child.read() {
                    if let NodeType::Trie(child_trie) = &*guard {
                        self.collect_pairs(child_trie, new_prefix, pairs);
                    }
                }
            }
        }
    }

    fn convert_to_btree(&self, trie_node: &TrieNode<K, V>) -> NodeType<K, V> {
        // Create a new B+ tree leaf node
        let mut btree_node = BTreeNode {
            keys: Vec::with_capacity(BTREE_MAX_KEYS),
            values: Vec::with_capacity(BTREE_MAX_KEYS),
            children: Vec::new(),
            is_leaf: true,
        };

        let mut pairs: Vec<(K, V)> = Vec::new();

        // Collect all key-value pairs from the trie
        self.collect_pairs(trie_node, Vec::new(), &mut pairs);

        // Sort pairs by key
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        // Fill the B+ tree node with sorted pairs
        for (key, value) in pairs {
            btree_node.keys.push(key);
            btree_node.values.push(value);
        }

        // Check if we need to split the node
        if btree_node.keys.len() > BTREE_MAX_KEYS {
            let mut root = BTreeNode {
                keys: Vec::new(),
                values: Vec::new(),
                children: Vec::new(),
                is_leaf: false,
            };

            // Split the oversized leaf node
            let split_info = self.split_btree_node(&mut btree_node).unwrap();

            // Create root node with two children
            root.keys.push(split_info.median_key);
            root.values.push(split_info.median_value);
            root.children
                .push(Arc::new(RwLock::new(NodeType::BTree(btree_node))));
            root.children.push(split_info.right_node);

            NodeType::BTree(root)
        } else {
            NodeType::BTree(btree_node)
        }
    }

    // Helper function to insert a key-value pair into the trie
    fn insert_into_trie(
        root: &mut TrieNode<K, V>,
        key: &K,
        value: V,
        current_depth: usize,
        key_bytes: &[u8],
    ) {
        if current_depth == key_bytes.len() {
            // We've reached the end of the key, store the value
            root.value = Some(value);
            root.size.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let current_byte = key_bytes[current_depth] as usize;

        // Create new node if it doesn't exist
        if root.children[current_byte].is_none() {
            root.children[current_byte] = Some(Arc::new(RwLock::new(NodeType::Trie(TrieNode {
                children: vec![None; 256],
                value: None,
                depth: current_depth + 1,
                size: AtomicUsize::new(0),
            }))));
        }

        // Get mutable reference to child node
        if let Some(child_arc) = &root.children[current_byte] {
            if let Ok(mut child_lock) = child_arc.write() {
                if let NodeType::Trie(child_trie) = &mut *child_lock {
                    Self::insert_into_trie(child_trie, key, value, current_depth + 1, key_bytes);
                }
            }
        }
    }

    // Helper function to collect leaf pairs from B+ tree
    fn collect_leaf_pairs(node: &BTreeNode<K, V>, pairs: &mut Vec<(K, V)>) {
        if node.is_leaf {
            pairs.extend(node.keys.iter().cloned().zip(node.values.iter().cloned()));
        } else {
            // Recursively collect from all children
            for child in &node.children {
                if let Ok(child_guard) = child.read() {
                    if let NodeType::BTree(child_btree) = &*child_guard {
                        Self::collect_leaf_pairs(child_btree, pairs);
                    }
                }
            }
        }
    }

    fn convert_to_trie(&self, btree_node: &BTreeNode<K, V>) -> NodeType<K, V> {
        // Create root trie node
        let mut root = TrieNode {
            children: vec![None; 256],
            value: None,
            depth: 0,
            size: AtomicUsize::new(0),
        };

        // If this is a leaf node, directly convert all key-value pairs
        if btree_node.is_leaf {
            for (key, value) in btree_node.keys.iter().zip(btree_node.values.iter()) {
                let key_bytes = key.to_bytes();
                Self::insert_into_trie(&mut root, key, value.clone(), 0, &key_bytes);
            }
        } else {
            let mut all_pairs = Vec::new();
            Self::collect_leaf_pairs(btree_node, &mut all_pairs);

            // Insert all collected pairs into the trie
            for (key, value) in all_pairs {
                let key_bytes = key.to_bytes();
                Self::insert_into_trie(&mut root, &key, value, 0, &key_bytes);
            }
        }

        NodeType::Trie(root)
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
