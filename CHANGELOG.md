# Changelog
All notable changes to the ASTrie project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-02-10

### Added
- Update functionality for existing key-value pairs
  - In-place value updates for both trie and B+ tree nodes
  - Returns previous value if key existed
  - Thread-safe implementation with proper locking
- Delete functionality for removing key-value pairs
  - Complete key-value pair removal
  - Returns removed value if key existed
  - Automatic node cleanup and merging
  - B+ tree rebalancing after deletion
- New helper methods for B+ tree maintenance
  - `merge_btree_nodes` for handling underflow
  - Node borrowing from siblings
  - Automatic node consolidation
- Comprehensive cleanup utilities
  - Empty node removal
  - Path cleanup after deletion
  - Memory optimization

### Changed
- Improved B+ tree node handling
  - Better sibling borrowing strategy
  - More efficient node merging
  - Optimized parent node updates
- Enhanced thread safety
  - Improved lock handling
  - Better concurrency support
  - Reduced lock contention

### Fixed
- Memory leaks in node deletion
- Lock handling in node merging
- Edge cases in B+ tree rebalancing

## [0.1.0] - 2025-02-01

### Added
- Initial release of ASTrie data structure
- Core insertion functionality
  - Support for both trie and B+ tree nodes
  - Automatic node type conversion
  - Thread-safe insertion
- Get operation implementation
  - Efficient key lookup
  - Support for both node types
  - Thread-safe retrieval
- Range query support
  - Inclusive range queries
  - Ordered result sets
  - Efficient traversal
- Key type abstractions
  - ToBytes trait for key serialization
  - FromBytes trait for key deserialization
  - Support for custom key types
- Thread safety features
  - Read-write locking
  - Atomic operations
  - Concurrent access support
- Node type conversion
  - Trie to B+ tree conversion
  - Adaptive node type selection
  - Performance-based switching

### Testing
- Unit tests for core operations
- Integration tests
- Concurrent operation tests
- Custom type tests
