//! Utility functions and helpers for the ASTrie data structure implementation.
//!
//! This module provides internal utility functions that support the core ASTrie operations:
//! - Node type conversions (trie ↔ B+ tree)
//! - Range query collection helpers
//! - Key-value pair collection utilities
//!
//! # Internal Use
//! These utilities are meant for internal use by the ASTrie implementation.
//! They handle the complex operations needed to maintain the adaptive behavior
//! of the data structure.
//!
//! # Thread Safety
//! All utility functions are designed to work with the ASTrie's concurrent
//! access model, properly handling read locks and shared references.
//!
//! # Performance Considerations
//! - Range collection operations are optimized for minimal memory allocation
//! - Node conversions maintain O(n) complexity where n is the number of key-value pairs
//! - Lock handling is designed to minimize contention
//!
//! # Error Handling
//! Utility functions use Option and Result types to handle error cases gracefully:
//! - Missing children in trie nodes
//! - Invalid byte sequences
//! - Lock acquisition failures
//!
//! # Notes
//! - All functions in this module are marked with `pub(crate)` visibility
//! - Helper functions maintain the invariants required by the main ASTrie implementation
//! - Documentation includes examples for internal use

use crate::{BTreeNode, FromBytes, NodeType, SplitInfo, ToBytes, TrieNode, BTREE_MAX_KEYS};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Helper method to collect range from trie
pub(crate) fn collect_trie_range<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &TrieNode<K, V>,
    prefix: Vec<u8>,
    start: &K,
    end: &K,
    result: &mut Vec<(K, V)>,
) {
    // If current node has a value, check if it falls within range
    if let Some(value) = &node.value {
        // Try to reconstruct key from prefix bytes
        if let Some(key) = K::from_bytes(&prefix) {
            // If key is within range, add key-value pair to results
            if &key >= start && &key <= end {
                result.push((key, value.clone()));
            }
        }
    }

    // Iterate through all possible child nodes
    for (byte, child_opt) in node.children.iter().enumerate() {
        if let Some(child) = child_opt {
            let mut new_prefix: Vec<u8> = prefix.clone();
            new_prefix.push(byte as u8);

            // Acquire read lock on child node
            if let Ok(child_guard) = child.read() {
                // If child is a trie node, recursively collect range
                if let NodeType::Trie(child_trie) = &*child_guard {
                    collect_trie_range(child_trie, new_prefix, start, end, result);
                }
            }
        }
    }
}

/// Helper method to collect range from B+ tree
pub(crate) fn collect_btree_range<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &BTreeNode<K, V>,
    start: &K,
    end: &K,
    result: &mut Vec<(K, V)>,
) {
    if node.is_leaf {
        // Find starting position using binary search
        let start_idx: usize = match node.keys.binary_search(start) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        // Collect all values from start_idx until we exceed end key
        for idx in start_idx..node.keys.len() {
            if &node.keys[idx] > end {
                break;
            }
            result.push((node.keys[idx].clone(), node.values[idx].clone()));
        }
    } else {
        // For internal nodes, find the child that might contain start key
        let mut current_idx: usize = match node.keys.binary_search(start) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };

        // Process relevant children until we exceed end key
        while current_idx < node.children.len() {
            if current_idx > 0 && &node.keys[current_idx - 1] > end {
                break;
            }

            // Process child node
            if let Ok(child_guard) = node.children[current_idx].read() {
                match &*child_guard {
                    NodeType::BTree(child_btree) => {
                        // Recursively collect from B+ tree child
                        collect_btree_range(child_btree, start, end, result);
                    }
                    NodeType::Trie(child_trie) => {
                        // Switch to trie collection for trie child
                        collect_trie_range(child_trie, Vec::new(), start, end, result);
                    }
                }
            }
            current_idx += 1;
        }
    }
}

/// Helper method to insert a key-value pair into the trie
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

/// Helper method to collect leaf pairs from B+ tree
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

/// Helper method to convert key to bytes for trie traversal
pub(crate) fn key_to_bytes<K: Clone + Ord + ToBytes + FromBytes>(key: &K) -> Vec<u8> {
    key.to_bytes()
}

/// Helper method to handle the split info and update parent node
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

/// Helper method to split a B+ tree node that has exceeded its maximum capacity
pub(crate) fn split_btree_node<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &mut BTreeNode<K, V>,
) -> Option<SplitInfo<K, V>> {
    let mid: usize = node.keys.len() / 2;

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
        let right_node: Arc<RwLock<NodeType<K, V>>> =
            Arc::new(RwLock::new(NodeType::BTree(right_node)));

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

/// Helper method to collect key-value pairs from trie
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

/// Helper method to convert a B+ tree node into a trie
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

/// Helper method to convert a trie node into a B+ tree
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

/// Helper method to merge underfull B+ tree nodes
pub fn merge_btree_nodes<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    node: &mut BTreeNode<K, V>,
    path: &[(Arc<RwLock<NodeType<K, V>>>, usize)],
) {
    if path.is_empty() {
        return;
    }

    let (parent_arc, idx) = path.last().unwrap();
    let mut parent = parent_arc.write().unwrap();

    if let NodeType::BTree(parent_node) = &mut *parent {
        // Try to borrow from siblings first
        if *idx > 0 {
            // Handle left sibling
            let can_borrow_left = {
                if let Ok(left_sibling) = parent_node.children[idx - 1].read() {
                    if let NodeType::BTree(left) = &*left_sibling {
                        left.keys.len() > BTREE_MAX_KEYS / 2
                    } else {
                        false
                    }
                } else {
                    false
                }
            };

            if can_borrow_left {
                // Rotate right
                if let Ok(mut left_sibling) = parent_node.children[idx - 1].write() {
                    if let NodeType::BTree(left) = &mut *left_sibling {
                        node.keys.insert(0, parent_node.keys[idx - 1].clone());
                        node.values.insert(0, parent_node.values[idx - 1].clone());
                        parent_node.keys[idx - 1] = left.keys.pop().unwrap();
                        parent_node.values[idx - 1] = left.values.pop().unwrap();
                        return;
                    }
                }
            }
        }

        if *idx < parent_node.children.len() - 1 {
            // Handle right sibling
            let can_borrow_right = {
                if let Ok(right_sibling) = parent_node.children[idx + 1].read() {
                    if let NodeType::BTree(right) = &*right_sibling {
                        right.keys.len() > BTREE_MAX_KEYS / 2
                    } else {
                        false
                    }
                } else {
                    false
                }
            };

            if can_borrow_right {
                // Rotate left
                if let Ok(mut right_sibling) = parent_node.children[idx + 1].write() {
                    if let NodeType::BTree(right) = &mut *right_sibling {
                        node.keys.push(parent_node.keys[*idx].clone());
                        node.values.push(parent_node.values[*idx].clone());
                        parent_node.keys[*idx] = right.keys.remove(0);
                        parent_node.values[*idx] = right.values.remove(0);
                        return;
                    }
                }
            }
        }

        // If we can't borrow, merge with a sibling
        if *idx > 0 {
            // Merge with left sibling
            let merged_node = {
                if let Ok(left_sibling) = parent_node.children[idx - 1].read() {
                    if let NodeType::BTree(left) = &*left_sibling {
                        let mut merged = BTreeNode {
                            keys: left.keys.clone(),
                            values: left.values.clone(),
                            children: left.children.clone(),
                            is_leaf: left.is_leaf,
                        };
                        merged.keys.push(parent_node.keys[idx - 1].clone());
                        merged.values.push(parent_node.values[idx - 1].clone());
                        merged.keys.extend(node.keys.iter().cloned());
                        merged.values.extend(node.values.iter().cloned());
                        merged.children.extend(node.children.iter().cloned());
                        Some(merged)
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            if let Some(merged) = merged_node {
                parent_node.children[idx - 1] = Arc::new(RwLock::new(NodeType::BTree(merged)));
                parent_node.keys.remove(idx - 1);
                parent_node.values.remove(idx - 1);
                parent_node.children.remove(*idx);
            }
        } else if *idx < parent_node.children.len() - 1 {
            // Merge with right sibling
            let merged_node = {
                if let Ok(right_sibling) = parent_node.children[idx + 1].read() {
                    if let NodeType::BTree(right) = &*right_sibling {
                        let mut merged = BTreeNode {
                            keys: node.keys.clone(),
                            values: node.values.clone(),
                            children: node.children.clone(),
                            is_leaf: node.is_leaf,
                        };
                        merged.keys.push(parent_node.keys[*idx].clone());
                        merged.values.push(parent_node.values[*idx].clone());
                        merged.keys.extend(right.keys.iter().cloned());
                        merged.values.extend(right.values.iter().cloned());
                        merged.children.extend(right.children.iter().cloned());
                        Some(merged)
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            if let Some(merged) = merged_node {
                parent_node.children[*idx] = Arc::new(RwLock::new(NodeType::BTree(merged)));
                parent_node.keys.remove(*idx);
                parent_node.values.remove(*idx);
                parent_node.children.remove(idx + 1);
            }
        }
    }
}

/// Helper method to clean up empty trie nodes after deletion
pub fn cleanup_empty_trie_nodes<K: Clone + Ord + ToBytes + FromBytes, V: Clone>(
    path: &[(Arc<RwLock<NodeType<K, V>>>, usize)],
) {
    for (node_arc, idx) in path.iter().rev() {
        let mut node = node_arc.write().unwrap();
        if let NodeType::Trie(trie_node) = &mut *node {
            if trie_node.children[*idx].as_ref().map_or(false, |child| {
                if let Ok(guard) = child.read() {
                    match &*guard {
                        NodeType::Trie(child_trie) => {
                            child_trie.value.is_none()
                                && child_trie.children.iter().all(|c| c.is_none())
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            }) {
                trie_node.children[*idx] = None;
            }
        }
    }
}
