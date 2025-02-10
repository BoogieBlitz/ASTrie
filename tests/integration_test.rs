use astrie::{ASTrie, FromBytes, ToBytes};
use std::sync::Arc;
use std::thread;

#[test]
fn test_basic_operations() {
    let trie: ASTrie<String, i32> = ASTrie::<String, i32>::new();

    // Test sequential inserts
    assert_eq!(trie.insert("apple".to_string(), 1), None);
    assert_eq!(trie.insert("banana".to_string(), 2), None);
    assert_eq!(trie.insert("cherry".to_string(), 3), None);

    // Test gets
    assert_eq!(trie.get(&"apple".to_string()), Some(1));
    assert_eq!(trie.get(&"banana".to_string()), Some(2));
    assert_eq!(trie.get(&"cherry".to_string()), Some(3));
    assert_eq!(trie.get(&"date".to_string()), None);

    // Test range query
    let range: Vec<(String, i32)> = trie.range(&"apple".to_string(), &"cherry".to_string());
    assert_eq!(range.len(), 3);
    assert_eq!(range[0], ("apple".to_string(), 1));
    assert_eq!(range[1], ("banana".to_string(), 2));
    assert_eq!(range[2], ("cherry".to_string(), 3));
}

#[test]
fn test_overwrite() {
    let trie: ASTrie<String, i32> = ASTrie::<String, i32>::new();

    // Insert and overwrite
    assert_eq!(trie.insert("key".to_string(), 1), None);
    assert_eq!(trie.insert("key".to_string(), 2), Some(1));
    assert_eq!(trie.insert("key".to_string(), 3), Some(2));

    // Verify final value
    assert_eq!(trie.get(&"key".to_string()), Some(3));
}

#[test]
fn test_empty_range() {
    let trie: ASTrie<String, i32> = ASTrie::<String, i32>::new();

    // Test range on empty trie
    assert!(trie.range(&"a".to_string(), &"z".to_string()).is_empty());

    // Test range with no matches
    trie.insert("key".to_string(), 1);
    assert!(trie.range(&"a".to_string(), &"b".to_string()).is_empty());
}

#[test]
fn test_concurrent_operations() {
    let trie: Arc<ASTrie<String, i32>> = Arc::new(ASTrie::<String, i32>::new());
    let mut handles: Vec<thread::JoinHandle<()>> = vec![];

    // Spawn multiple threads for insertion
    for i in 0..10 {
        let trie_clone = Arc::clone(&trie);
        let handle = thread::spawn(move || {
            let key = format!("key{}", i);
            trie_clone.insert(key, i);
        });
        handles.push(handle);
    }

    // Wait for all insertions to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all insertions
    for i in 0..10 {
        let key = format!("key{}", i);
        assert_eq!(trie.get(&key), Some(i));
    }

    // Test range query after concurrent insertions
    let range: Vec<(String, i32)> = trie.range(&"key0".to_string(), &"key9".to_string());
    assert_eq!(range.len(), 10);
}

#[test]
fn test_large_dataset() {
    let trie: ASTrie<i32, String> = ASTrie::<i32, String>::new();

    // Insert a large number of items
    for i in 0..1000 {
        trie.insert(i, i.to_string());
    }

    // Test random access
    assert_eq!(trie.get(&42), Some("42".to_string()));
    assert_eq!(trie.get(&999), Some("999".to_string()));
    assert_eq!(trie.get(&1000), None);

    // Test range query with large range
    let range: Vec<(i32, String)> = trie.range(&100, &200);
    assert_eq!(range.len(), 101);
    assert_eq!(range[0].0, 100);
    assert_eq!(range[100].0, 200);
}

// Test custom type
#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
struct CustomKey {
    id: u32,
    name: String,
}

impl ToBytes for CustomKey {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&self.id.to_be_bytes());
        bytes.extend_from_slice(self.name.as_bytes());
        bytes
    }
}

impl FromBytes for CustomKey {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let id: u32 = u32::from_be_bytes(bytes[0..4].try_into().ok()?);
        let name: String = String::from_utf8(bytes[4..].to_vec()).ok()?;
        Some(CustomKey { id, name })
    }
}

#[test]
fn test_custom_type() {
    let trie: ASTrie<CustomKey, String> = ASTrie::<CustomKey, String>::new();

    // Insert items with custom key
    let key1: CustomKey = CustomKey {
        id: 1,
        name: "first".to_string(),
    };
    let key2: CustomKey = CustomKey {
        id: 2,
        name: "second".to_string(),
    };

    trie.insert(key1.clone(), "value1".to_string());
    trie.insert(key2.clone(), "value2".to_string());

    // Test retrieval
    assert_eq!(trie.get(&key1), Some("value1".to_string()));

    // Test range query
    let range: Vec<(CustomKey, String)> = trie.range(&key1, &key2);
    assert_eq!(range.len(), 1);
    assert_eq!(range[0].1, "value2".to_string());
}

#[test]
fn test_trie_to_btree_conversion() {
    let trie: ASTrie<String, i32> = ASTrie::<String, i32>::new();

    // Insert enough items to trigger conversion to B+ tree
    for i in 0..1000 {
        trie.insert(format!("key{:03}", i), i as i32);
    }

    // Verify all items are still accessible
    for i in 0..1000 {
        assert_eq!(trie.get(&format!("key{:03}", i)), Some(i as i32));
    }

    // Test range query after conversion
    let range: Vec<(String, i32)> = trie.range(&"key000".to_string(), &"key009".to_string());
    assert_eq!(range.len(), 10);
}

#[test]
fn test_update() {
    let trie: ASTrie<String, i32> = ASTrie::<String, i32>::new();

    // Insert initial values
    trie.insert("key1".to_string(), 1);
    trie.insert("key2".to_string(), 2);

    // Test update existing key
    assert_eq!(trie.update(&"key1".to_string(), 10), Some(1));
    assert_eq!(trie.get(&"key1".to_string()), Some(10));

    // Test update non-existent key
    assert_eq!(trie.update(&"key3".to_string(), 3), None);
}

#[test]
fn test_delete() {
    let trie: ASTrie<String, i32> = ASTrie::<String, i32>::new();

    // Insert values
    trie.insert("key1".to_string(), 1);
    trie.insert("key2".to_string(), 2);
    trie.insert("key3".to_string(), 3);

    // Test delete existing key
    assert_eq!(trie.delete(&"key2".to_string()), Some(2));
    assert_eq!(trie.get(&"key2".to_string()), None);

    // Test delete non-existent key
    assert_eq!(trie.delete(&"key4".to_string()), None);

    // Verify remaining keys
    assert_eq!(trie.get(&"key1".to_string()), Some(1));
    assert_eq!(trie.get(&"key3".to_string()), Some(3));
}

#[test]
fn test_concurrent_update_delete() {
    let trie: Arc<ASTrie<String, i32>> = Arc::new(ASTrie::<String, i32>::new());

    // Insert initial values
    for i in 0..100 {
        trie.insert(format!("key{}", i), i);
    }

    let trie_clone: Arc<ASTrie<String, i32>> = Arc::clone(&trie);
    let update_thread = thread::spawn(move || {
        for i in 0..50 {
            trie_clone.update(&format!("key{}", i), i * 10);
        }
    });

    let trie_clone: Arc<ASTrie<String, i32>> = Arc::clone(&trie);
    let delete_thread = thread::spawn(move || {
        for i in 50..100 {
            trie_clone.delete(&format!("key{}", i));
        }
    });

    update_thread.join().unwrap();
    delete_thread.join().unwrap();

    // Verify results
    for i in 0..50 {
        assert_eq!(trie.get(&format!("key{}", i)), Some(i * 10));
    }
    for i in 50..100 {
        assert_eq!(trie.get(&format!("key{}", i)), None);
    }
}

#[test]
fn test_btree_node_merging() {
    let trie: ASTrie<i32, String> = ASTrie::<i32, String>::new();

    // Insert enough values to create B+ tree nodes
    for i in 0..1000 {
        trie.insert(i, i.to_string());
    }

    // Delete many values to force node merging
    for i in (0..900).step_by(2) {
        assert_eq!(trie.delete(&i), Some(i.to_string()));
    }

    // Verify remaining values
    for i in (1..900).step_by(2) {
        assert_eq!(trie.get(&i), Some(i.to_string()));
    }
}

#[test]
fn test_edge_cases() {
    let trie: ASTrie<String, i32> = ASTrie::<String, i32>::new();

    // Test empty string
    trie.insert("".to_string(), 1);
    assert_eq!(trie.update(&"".to_string(), 2), Some(1));
    assert_eq!(trie.delete(&"".to_string()), Some(2));

    // Test very long key
    let long_key: String = "a".repeat(1000);
    trie.insert(long_key.clone(), 1);
    assert_eq!(trie.update(&long_key, 2), Some(1));
    assert_eq!(trie.delete(&long_key), Some(2));

    // Test Unicode
    trie.insert("🦀".to_string(), 1);
    assert_eq!(trie.update(&"🦀".to_string(), 2), Some(1));
    assert_eq!(trie.delete(&"🦀".to_string()), Some(2));
}
