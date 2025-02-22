// benches/benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use astrie::ASTrie;
use std::collections::BTreeMap;
use rand::{thread_rng, Rng};

// Reduce test data sizes
const SMALL_SIZE: usize = 100;
const MEDIUM_SIZE: usize = 1000;
const LARGE_SIZE: usize = 5000;

fn generate_random_string(len: usize) -> String {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = thread_rng();
    (0..len)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

fn generate_sequential_string(n: usize) -> String {
    format!("key{:05}", n) // Reduced padding size
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("Insert Operations");
    group.sample_size(10); // Reduce sample size
    
    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    for size in sizes {
        // Random strings
        group.bench_with_input(BenchmarkId::new("ASTrie/Random", size), &size, |b, &size| {
            b.iter_batched(
                || (), // Setup
                |_| {
                    let trie = ASTrie::<String, i32>::new();
                    for i in 0..size {
                        let key = generate_random_string(5); // Reduced string length
                        trie.insert(key, i as i32);
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap/Random", size), &size, |b, &size| {
            b.iter_batched(
                || (),
                |_| {
                    let mut btree = BTreeMap::new();
                    for i in 0..size {
                        let key = generate_random_string(5);
                        btree.insert(key, i as i32);
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("Get Operations");
    group.sample_size(10);
    
    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("ASTrie/Get", size), &size, |b, &size| {
            // Setup phase
            let trie = ASTrie::<String, i32>::new();
            let mut keys = Vec::with_capacity(size);
            
            for i in 0..size {
                let key = generate_sequential_string(i);
                keys.push(key.clone());
                trie.insert(key, i as i32);
            }

            // Benchmark phase
            b.iter(|| {
                let mut rng = thread_rng();
                let idx = rng.gen_range(0..size);
                black_box(trie.get(&keys[idx]));
            });
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap/Get", size), &size, |b, &size| {
            let mut btree = BTreeMap::new();
            let mut keys = Vec::with_capacity(size);
            
            for i in 0..size {
                let key = generate_sequential_string(i);
                keys.push(key.clone());
                btree.insert(key, i as i32);
            }

            b.iter(|| {
                let mut rng = thread_rng();
                let idx = rng.gen_range(0..size);
                black_box(btree.get(&keys[idx]));
            });
        });
    }
    group.finish();
}

fn bench_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("Range Operations");
    group.sample_size(10);
    
    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("ASTrie/Range", size), &size, |b, &size| {
            // Setup
            let trie = ASTrie::<String, i32>::new();
            for i in 0..size {
                let key = generate_sequential_string(i);
                trie.insert(key, i as i32);
            }

            // Benchmark
            b.iter(|| {
                let start = generate_sequential_string(size / 4);
                let end = generate_sequential_string(size / 2);
                black_box(trie.range(&start, &end));
            });
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap/Range", size), &size, |b, &size| {
            let mut btree = BTreeMap::new();
            for i in 0..size {
                let key = generate_sequential_string(i);
                btree.insert(key, i as i32);
            }

            b.iter(|| {
                let start = generate_sequential_string(size / 4);
                let end = generate_sequential_string(size / 2);
                black_box(btree.range(start..=end).collect::<Vec<_>>());
            });
        });
    }
    group.finish();
}


fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("Update Operations");
    group.sample_size(10);
    
    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("ASTrie/Update", size), &size, |b, &size| {
            // Setup phase
            let trie = ASTrie::<String, i32>::new();
            let mut keys = Vec::with_capacity(size);
            
            // Insert initial data
            for i in 0..size {
                let key = generate_sequential_string(i);
                keys.push(key.clone());
                trie.insert(key, i as i32);
            }

            // Benchmark phase - update existing keys
            let mut rng = thread_rng();
            b.iter(|| {
                let idx = rng.gen_range(0..size);
                let new_value = rng.gen::<i32>();
                black_box(trie.update(&keys[idx], new_value));
            });
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap/Update", size), &size, |b, &size| {
            let mut btree = BTreeMap::new();
            let mut keys = Vec::with_capacity(size);
            
            for i in 0..size {
                let key = generate_sequential_string(i);
                keys.push(key.clone());
                btree.insert(key, i as i32);
            }

            let mut rng = thread_rng();
            b.iter(|| {
                let idx = rng.gen_range(0..size);
                let new_value = rng.gen::<i32>();
                black_box(btree.insert(keys[idx].clone(), new_value));
            });
        });
    }
    group.finish();
}

fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("Delete Operations");
    group.sample_size(10);
    
    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("ASTrie/Delete", size), &size, |b, &size| {
            b.iter_batched(
                // Setup: Create new trie and insert data
                || {
                    let trie = ASTrie::<String, i32>::new();
                    let mut keys = Vec::with_capacity(size);
                    for i in 0..size {
                        let key = generate_sequential_string(i);
                        keys.push(key.clone());
                        trie.insert(key, i as i32);
                    }
                    (trie, keys)
                },
                // Benchmark: Delete random keys
                |(trie, keys)| {
                    let mut rng = thread_rng();
                    let idx = rng.gen_range(0..keys.len());
                    black_box(trie.delete(&keys[idx]));
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap/Delete", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut btree = BTreeMap::new();
                    let mut keys = Vec::with_capacity(size);
                    for i in 0..size {
                        let key = generate_sequential_string(i);
                        keys.push(key.clone());
                        btree.insert(key, i as i32);
                    }
                    (btree, keys)
                },
                |(mut btree, keys)| {
                    let mut rng = thread_rng();
                    let idx = rng.gen_range(0..keys.len());
                    black_box(btree.remove(&keys[idx]));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// Add a benchmark for mixed operations
fn bench_mixed_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed Operations");
    group.sample_size(10);
    
    let sizes = [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("ASTrie/Mixed", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let trie = ASTrie::<String, i32>::new();
                    let mut keys = Vec::with_capacity(size);
                    for i in 0..size {
                        let key = generate_sequential_string(i);
                        keys.push(key.clone());
                        trie.insert(key, i as i32);
                    }
                    (trie, keys)
                },
                |(trie, keys)| {
                    let mut rng = thread_rng();
                    match rng.gen_range(0..4) {
                        0 => { // Insert
                            let key = generate_random_string(5);
                            black_box(trie.insert(key, rng.gen()));
                        },
                        1 => { // Update
                            let idx = rng.gen_range(0..keys.len());
                            black_box(trie.update(&keys[idx], rng.gen()));
                        },
                        2 => { // Delete
                            let idx = rng.gen_range(0..keys.len());
                            black_box(trie.delete(&keys[idx]));
                        },
                        3 => { // Get
                            let idx = rng.gen_range(0..keys.len());
                            black_box(trie.get(&keys[idx]));
                        },
                        _ => unreachable!(),
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap/Mixed", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut btree = BTreeMap::new();
                    let mut keys = Vec::with_capacity(size);
                    for i in 0..size {
                        let key = generate_sequential_string(i);
                        keys.push(key.clone());
                        btree.insert(key, i as i32);
                    }
                    (btree, keys)
                },
                |(mut btree, keys)| {
                    let mut rng = thread_rng();
                    match rng.gen_range(0..4) {
                        0 => { // Insert
                            let key = generate_random_string(5);
                            black_box(btree.insert(key, rng.gen()));
                        },
                        1 => { // Update
                            let idx = rng.gen_range(0..keys.len());
                            black_box(btree.insert(keys[idx].clone(), rng.gen()));
                        },
                        2 => { // Delete
                            let idx = rng.gen_range(0..keys.len());
                            black_box(btree.remove(&keys[idx]));
                        },
                        3 => { // Get
                            let idx = rng.gen_range(0..keys.len());
                            black_box(btree.get(&keys[idx]));
                        },
                        _ => unreachable!(),
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_insert, bench_get, bench_range, 
             bench_update, bench_delete, bench_mixed_operations
}
criterion_main!(benches);
