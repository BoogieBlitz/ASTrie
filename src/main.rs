use astrie::ASTrie;

fn main() {
    // Example 1: Basic String Key Usage
    println!("Example 1: String Keys");
    let string_trie: ASTrie<String, i32> = ASTrie::new();

    // Insert some data
    string_trie.insert("hello".to_string(), 1);
    string_trie.insert("help".to_string(), 2);
    string_trie.insert("world".to_string(), 3);

    // Lookup
    println!("hello -> {:?}", string_trie.get(&"hello".to_string()));
    println!("help -> {:?}", string_trie.get(&"help".to_string()));
    println!("missing -> {:?}", string_trie.get(&"missing".to_string()));
    assert_eq!(string_trie.get(&"hello".to_string()), Some(1));
    assert_eq!(string_trie.get(&"help".to_string()), Some(2));
    assert_eq!(string_trie.get(&"missing".to_string()), None);
}
