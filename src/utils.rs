use crate::{BTreeNode, FromBytes, NodeType, SplitInfo, ToBytes, TrieNode, BTREE_MAX_KEYS};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

// Helper function to collect range from trie
pub(crate) fn collect_trie_range<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &TrieNode<K, V>,
    prefix: Vec<u8>,
    start: &K,
    end: &K,
    result: &mut Vec<(K, V)>,
) {
    if let Some(value) = &node.value {
        if let Some(key) = K::from_bytes(&prefix) {
            if &key >= start && &key <= end {
                result.push((key, value.clone()));
            }
        }
    }

    for (byte, child_opt) in node.children.iter().enumerate() {
        if let Some(child) = child_opt {
            let mut new_prefix = prefix.clone();
            new_prefix.push(byte as u8);

            if let Ok(child_guard) = child.read() {
                if let NodeType::Trie(child_trie) = &*child_guard {
                    collect_trie_range(child_trie, new_prefix, start, end, result);
                }
            }
        }
    }
}

// Helper function to collect range from B+ tree
pub(crate) fn collect_btree_range<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &BTreeNode<K, V>,
    start: &K,
    end: &K,
    result: &mut Vec<(K, V)>,
) {
    if node.is_leaf {
        let start_idx = match node.keys.binary_search(start) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        for idx in start_idx..node.keys.len() {
            if &node.keys[idx] > end {
                break;
            }
            result.push((node.keys[idx].clone(), node.values[idx].clone()));
        }
    } else {
        let mut current_idx = match node.keys.binary_search(start) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };

        while current_idx < node.children.len() {
            if current_idx > 0 && &node.keys[current_idx - 1] > end {
                break;
            }

            if let Ok(child_guard) = node.children[current_idx].read() {
                match &*child_guard {
                    NodeType::BTree(child_btree) => {
                        collect_btree_range(child_btree, start, end, result);
                    }
                    NodeType::Trie(child_trie) => {
                        collect_trie_range(child_trie, Vec::new(), start, end, result);
                    }
                }
            }
            current_idx += 1;
        }
    }
}

// Helper function to insert a key-value pair into the trie
pub(crate) fn insert_into_trie<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
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
                insert_into_trie(child_trie, key, value, current_depth + 1, key_bytes);
            }
        }
    }
}

// Helper function to collect leaf pairs from B+ tree
pub(crate) fn collect_leaf_pairs<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &BTreeNode<K, V>,
    pairs: &mut Vec<(K, V)>,
) {
    if node.is_leaf {
        pairs.extend(node.keys.iter().cloned().zip(node.values.iter().cloned()));
    } else {
        // Recursively collect from all children
        for child in &node.children {
            if let Ok(child_guard) = child.read() {
                if let NodeType::BTree(child_btree) = &*child_guard {
                    collect_leaf_pairs(child_btree, pairs);
                }
            }
        }
    }
}

// Helper method to convert key to bytes for trie traversal
pub(crate) fn key_to_bytes<K: Clone + Ord + ToBytes + FromBytes>(key: &K) -> Vec<u8> {
    key.to_bytes()
}

// Helper method to handle the split info and update parent node
pub(crate) fn handle_split<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
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

pub(crate) fn split_btree_node<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &mut BTreeNode<K, V>,
) -> Option<SplitInfo<K, V>> {
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
pub(crate) fn collect_pairs<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &TrieNode<K, V>,
    prefix: Vec<u8>,
    pairs: &mut Vec<(K, V)>,
) {
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
                    collect_pairs(child_trie, new_prefix, pairs);
                }
            }
        }
    }
}

pub(crate) fn convert_to_trie<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    btree_node: &BTreeNode<K, V>,
) -> NodeType<K, V> {
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
            insert_into_trie(&mut root, key, value.clone(), 0, &key_bytes);
        }
    } else {
        let mut all_pairs: Vec<(K, V)> = Vec::new();
        collect_leaf_pairs(btree_node, &mut all_pairs);

        // Insert all collected pairs into the trie
        for (key, value) in all_pairs {
            let key_bytes: Vec<u8> = key.to_bytes();
            insert_into_trie(&mut root, &key, value, 0, &key_bytes);
        }
    }

    NodeType::Trie(root)
}

pub(crate) fn convert_to_btree<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    trie_node: &TrieNode<K, V>,
) -> NodeType<K, V> {
    // Create a new B+ tree leaf node
    let mut btree_node = BTreeNode {
        keys: Vec::with_capacity(BTREE_MAX_KEYS),
        values: Vec::with_capacity(BTREE_MAX_KEYS),
        children: Vec::new(),
        is_leaf: true,
    };

    let mut pairs: Vec<(K, V)> = Vec::new();

    // Collect all key-value pairs from the trie
    collect_pairs(trie_node, Vec::new(), &mut pairs);

    // Sort pairs by key
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // Fill the B+ tree node with sorted pairs
    for (key, value) in pairs {
        btree_node.keys.push(key);
        btree_node.values.push(value);
    }

    // Check if we need to split the node
    if btree_node.keys.len() > BTREE_MAX_KEYS {
        let mut root: BTreeNode<K, V> = BTreeNode {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: false,
        };

        // Split the oversized leaf node
        let split_info: SplitInfo<K, V> = split_btree_node(&mut btree_node).unwrap();

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
