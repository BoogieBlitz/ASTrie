use astrie::{ASTrie, ToBytes, FromBytes};
use std::time::Instant;

// Custom type for demonstrating complex keys
#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
struct UserProfile {
    id: u32,
    username: String,
}

// Implement serialization for our custom type
impl ToBytes for UserProfile {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.id.to_be_bytes());
        bytes.extend_from_slice(self.username.as_bytes());
        bytes
    }
}

impl FromBytes for UserProfile {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let id = u32::from_be_bytes(bytes[0..4].try_into().ok()?);
        let username = String::from_utf8(bytes[4..].to_vec()).ok()?;
        Some(UserProfile { id, username })
    }
}

fn main() {
    // Example 1: Basic String key usage
    println!("\n=== Example 1: Basic String Keys ===");
    let string_trie = ASTrie::<String, i32>::new();
    
    // Insert operations
    let start: Instant = Instant::now();
    string_trie.insert("apple".to_string(), 1);
    string_trie.insert("banana".to_string(), 2);
    string_trie.insert("cherry".to_string(), 3);
    println!("Inserted 3 items in {:?}", start.elapsed());

    // Get operations
    println!("Value for 'apple': {:?}", string_trie.get(&"apple".to_string()));
    println!("Value for 'missing': {:?}", string_trie.get(&"missing".to_string()));

    // Update operation
    println!("Old value when updating 'apple': {:?}", 
             string_trie.update(&"apple".to_string(), 10));
    println!("New value for 'apple': {:?}", 
             string_trie.get(&"apple".to_string()));

    // Range query
    let range: Vec<(String, i32)> = string_trie.range(&"apple".to_string(), &"cherry".to_string());
    println!("Range query results (apple to cherry):");
    for (key, value) in range {
        println!("  {} -> {}", key, value);
    }

    // Delete operation
    println!("Deleted value for 'banana': {:?}", 
             string_trie.delete(&"banana".to_string()));
    println!("Try get 'banana' after delete: {:?}", 
             string_trie.get(&"banana".to_string()));

    // Example 2: Custom type (UserProfile) usage
    println!("\n=== Example 2: Custom Type (UserProfile) ===");
    let user_trie: ASTrie<UserProfile, String> = ASTrie::<UserProfile, String>::new();

    // Create some user profiles
    let user1: UserProfile = UserProfile { id: 1, username: "alice".to_string() };
    let user2: UserProfile = UserProfile { id: 2, username: "bob".to_string() };
    let user3: UserProfile = UserProfile { id: 3, username: "charlie".to_string() };

    // Insert users
    user_trie.insert(user1.clone(), "Alice's data".to_string());
    user_trie.insert(user2.clone(), "Bob's data".to_string());
    user_trie.insert(user3.clone(), "Charlie's data".to_string());

    // Get user data
    println!("Data for user 1: {:?}", user_trie.get(&user1));

    // Update user data
    user_trie.update(&user2, "Bob's updated data".to_string());
    println!("Updated data for user 2: {:?}", user_trie.get(&user2));

    // Range query for users
    println!("Users from id 1 to 2:");
    for (user, data) in user_trie.range(&user1, &user2) {
        println!("  User {} ({}) -> {}", user.id, user.username, data);
    }

    // Example 3: Numeric keys with bulk operations
    println!("\n=== Example 3: Bulk Operations ===");
    let numeric_trie: ASTrie<i32, String> = ASTrie::<i32, String>::new();

    // Bulk insert
    let start: Instant = Instant::now();
    for i in 0..1000 {
        numeric_trie.insert(i, format!("Value {}", i));
    }
    println!("Inserted 1000 items in {:?}", start.elapsed());

    // Bulk update
    let start: Instant = Instant::now();
    for i in 0..500 {
        numeric_trie.update(&i, format!("Updated value {}", i));
    }
    println!("Updated 500 items in {:?}", start.elapsed());

    // Range query with pagination
    let page_size: usize = 10;
    println!("First {} items in range 100-200:", page_size);
    let range = numeric_trie.range(&100, &200);
    for (key, value) in range.iter().take(page_size) {
        println!("  {} -> {}", key, value);
    }

    // Bulk delete
    let start: Instant = Instant::now();
    for i in 0..100 {
        numeric_trie.delete(&i);
    }
    println!("Deleted 100 items in {:?}", start.elapsed());

    // Example 4: Error handling and edge cases
    println!("\n=== Example 4: Error Handling and Edge Cases ===");
    let edge_trie: ASTrie<String, i32> = ASTrie::<String, i32>::new();

    // Empty string key
    edge_trie.insert("".to_string(), 0);
    println!("Empty string value: {:?}", edge_trie.get(&"".to_string()));

    // Very long key
    let long_key: String = "x".repeat(1000);
    edge_trie.insert(long_key.clone(), 1000);
    println!("Long key exists: {}", edge_trie.get(&long_key).is_some());

    // Unicode characters
    edge_trie.insert("🦀".to_string(), 42);
    println!("Unicode key value: {:?}", edge_trie.get(&"🦀".to_string()));

    // Update non-existent key
    println!("Update missing key: {:?}", 
             edge_trie.update(&"missing".to_string(), 404));

    // Delete non-existent key
    println!("Delete missing key: {:?}", 
             edge_trie.delete(&"missing".to_string()));

    // Empty range query
    let empty_range: Vec<(String, i32)> = edge_trie.range(&"z".to_string(), &"a".to_string());
    println!("Empty range query length: {}", empty_range.len());
}