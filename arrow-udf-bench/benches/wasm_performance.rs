// Copyright 2024 RisingWave Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    fs::{self, File},
    hint::black_box,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Float64Array, Int64Array, RecordBatch, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use arrow_udf_runtime::wasm::{Config, Runtime};

const BINARY_PATH: &str = "../target/wasm32-wasip1/release/arrow_udf_example.wasm";
const WARMUP: usize = 5;
const REPETITIONS: usize = 20;
const BATCH_SIZES: [usize; 6] = [1, 16, 256, 4_096, 16_384, 65_536];
const CONCURRENCY: [usize; 6] = [1, 2, 4, 8, 16, 32];

#[derive(Debug, Clone, Copy)]
struct Stats {
    median_ns: f64,
    mean_ns: f64,
    p95_ns: f64,
    p99_ns: f64,
}

impl Stats {
    fn from_samples(samples: &[u128]) -> Self {
        assert!(!samples.is_empty());
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let percentile = |fraction: f64| {
            let index = ((sorted.len() as f64 * fraction).ceil() as usize)
                .saturating_sub(1)
                .min(sorted.len() - 1);
            sorted[index] as f64
        };
        Self {
            median_ns: percentile(0.50),
            mean_ns: sorted.iter().sum::<u128>() as f64 / sorted.len() as f64,
            p95_ns: percentile(0.95),
            p99_ns: percentile(0.99),
        }
    }
}

struct Csv {
    writer: BufWriter<File>,
}

struct Row<'a> {
    layer: &'a str,
    category: &'a str,
    path: &'a str,
    value_type: &'a str,
    rows: usize,
    arguments: usize,
    concurrency: usize,
    partial_states: usize,
    state_bytes: usize,
    input_bytes: usize,
    output_bytes: usize,
    invocations: usize,
    errors: usize,
    samples: usize,
    stats: Stats,
}

impl Csv {
    fn create(path: &Path) -> Self {
        let mut writer = BufWriter::new(File::create(path).unwrap());
        writeln!(
            writer,
            "layer,category,path,type,rows,arguments,concurrency,partial_states,state_bytes,input_bytes,output_bytes,invocations,errors,warmup,repetitions,median_ns,mean_ns,p95_ns,p99_ns,ns_per_row,rows_per_second"
        )
        .unwrap();
        Self { writer }
    }

    fn row(&mut self, row: Row<'_>) {
        let ns_per_row = if row.rows == 0 {
            0.0
        } else {
            row.stats.median_ns / row.rows as f64
        };
        let rows_per_second = if row.rows == 0 || row.stats.median_ns == 0.0 {
            0.0
        } else {
            row.rows as f64 * 1_000_000_000.0 / row.stats.median_ns
        };
        writeln!(
            self.writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            row.layer,
            row.category,
            row.path,
            row.value_type,
            row.rows,
            row.arguments,
            row.concurrency,
            row.partial_states,
            row.state_bytes,
            row.input_bytes,
            row.output_bytes,
            row.invocations,
            row.errors,
            WARMUP,
            row.samples,
            row.stats.median_ns,
            row.stats.mean_ns,
            row.stats.p95_ns,
            row.stats.p99_ns,
            ns_per_row,
            rows_per_second,
        )
        .unwrap();
    }
}

fn output_directory() -> PathBuf {
    std::env::var_os("WASM_BENCH_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/wasm-performance"))
}

fn runtime_config() -> Config {
    let mut config = Config::default();
    config.memory_size_limit = Some(64 * 1024 * 1024);
    config.table_elements_limit = Some(100_000);
    config.instance_limit = Some(8);
    config.table_limit = Some(8);
    config.memory_count_limit = Some(4);
    config.fuel_per_call = Some(100_000_000);
    config.epoch_deadline_ticks = Some(200);
    config.file_size_limit = Some(4 * 1024);
    config
}

fn runtime(binary: &[u8]) -> Runtime {
    runtime_with_pool(binary, None)
}

fn runtime_with_pool(binary: &[u8], isolated_instance_pool_size: Option<usize>) -> Runtime {
    let mut config = runtime_config();
    config.isolated_instance_pool_size = isolated_instance_pool_size;
    Runtime::builder(binary).config(config).build().unwrap()
}

fn measure<T>(mut operation: impl FnMut() -> T) -> Stats {
    for _ in 0..WARMUP {
        black_box(operation());
    }
    let mut samples = Vec::with_capacity(REPETITIONS);
    for _ in 0..REPETITIONS {
        let start = Instant::now();
        black_box(operation());
        samples.push(start.elapsed().as_nanos());
    }
    Stats::from_samples(&samples)
}

fn int64_batch(rows: usize, arguments: usize) -> RecordBatch {
    let columns = (0..arguments)
        .map(|argument| {
            Arc::new(Int64Array::from_iter_values(
                (0..rows).map(|row| row as i64 + argument as i64),
            )) as ArrayRef
        })
        .collect::<Vec<_>>();
    let fields = (0..arguments)
        .map(|argument| Field::new(format!("arg{argument}"), DataType::Int64, false))
        .collect::<Vec<_>>();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn float64_batch(rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )])),
        vec![Arc::new(Float64Array::from_iter(
            (0..rows).map(|row| (row % 19 != 0).then_some(row as f64 * 0.25)),
        ))],
    )
    .unwrap()
}

fn uint64_batch(rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )])),
        vec![Arc::new(UInt64Array::from_iter_values(0..rows as u64))],
    )
    .unwrap()
}

fn boolean_batch(rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Boolean,
            false,
        )])),
        vec![Arc::new(BooleanArray::from(
            (0..rows).map(|row| row % 2 == 0).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn native_scalar(batch: &RecordBatch, value_type: &str, arguments: usize) -> ArrayRef {
    match value_type {
        "Int64" => {
            let arrays = batch
                .columns()
                .iter()
                .map(|array| array.as_any().downcast_ref::<Int64Array>().unwrap())
                .collect::<Vec<_>>();
            Arc::new(Int64Array::from_iter_values((0..batch.num_rows()).map(
                |row| {
                    arrays.iter().map(|array| array.value(row)).sum::<i64>()
                        + (arguments == 1) as i64
                },
            )))
        }
        "UInt64" => {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            Arc::new(UInt64Array::from_iter_values(
                values.values().iter().map(|value| value + 1),
            ))
        }
        "Float64" => {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            Arc::new(Float64Array::from_iter(
                values.iter().map(|value| value.map(|value| value * 2.0)),
            ))
        }
        "Boolean" => {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            Arc::new(BooleanArray::from_iter(
                values.iter().map(|value| value.map(|value| !value)),
            ))
        }
        _ => unreachable!(),
    }
}

fn scalar_case(
    csv: &mut Csv,
    runtime: &Runtime,
    value_type: &'static str,
    data_type: DataType,
    export: &'static str,
    batch: RecordBatch,
    arguments: usize,
) {
    let function = runtime
        .find_function(
            export,
            vec![data_type.clone(); arguments],
            data_type.clone(),
        )
        .unwrap();
    let expected = native_scalar(&batch, value_type, arguments);
    let actual = runtime.call(&function, &batch).unwrap();
    assert_eq!(actual.column(0).to_data(), expected.to_data());
    let native = measure(|| native_scalar(black_box(&batch), value_type, arguments));
    let wasm = measure(|| runtime.call(&function, black_box(&batch)).unwrap());
    let input_bytes = batch.get_array_memory_size();
    let output_bytes = actual.get_array_memory_size();
    for (path, stats) in [("native_arrow", native), ("wasm", wasm)] {
        csv.row(Row {
            layer: "A",
            category: "scalar_batch",
            path,
            value_type,
            rows: batch.num_rows(),
            arguments,
            concurrency: 1,
            partial_states: 0,
            state_bytes: 0,
            input_bytes,
            output_bytes,
            invocations: 1,
            errors: 0,
            samples: REPETITIONS,
            stats,
        });
    }
}

fn scalar_matrices(csv: &mut Csv, runtime: &Runtime) {
    for rows in BATCH_SIZES {
        scalar_case(
            csv,
            runtime,
            "Int64",
            DataType::Int64,
            "add_one_int64",
            int64_batch(rows, 1),
            1,
        );
    }
    for (value_type, data_type, export, batch) in [
        (
            "UInt64",
            DataType::UInt64,
            "add_one_uint64",
            uint64_batch(4_096),
        ),
        (
            "Float64",
            DataType::Float64,
            "double_float64",
            float64_batch(4_096),
        ),
        (
            "Boolean",
            DataType::Boolean,
            "boolean_not",
            boolean_batch(4_096),
        ),
    ] {
        scalar_case(csv, runtime, value_type, data_type, export, batch, 1);
    }
    for (arguments, export) in [(2, "bench_add2_int64"), (4, "bench_add4_int64")] {
        scalar_case(
            csv,
            runtime,
            "Int64",
            DataType::Int64,
            export,
            int64_batch(4_096, arguments),
            arguments,
        );
    }
}

fn cold_warm(csv: &mut Csv, binary: &[u8]) {
    let compile = measure(|| runtime(black_box(binary)));
    csv.row(Row {
        layer: "A",
        category: "scalar_cold_warm",
        path: "module_compile",
        value_type: "Int64",
        rows: 0,
        arguments: 1,
        concurrency: 1,
        partial_states: 0,
        state_bytes: 0,
        input_bytes: binary.len(),
        output_bytes: 0,
        invocations: 0,
        errors: 0,
        samples: REPETITIONS,
        stats: compile,
    });

    let runtime = runtime(binary);
    let lookup = measure(|| {
        runtime
            .find_function("add_one_int64", vec![DataType::Int64], DataType::Int64)
            .unwrap()
    });
    csv.row(Row {
        layer: "A",
        category: "scalar_cold_warm",
        path: "function_lookup",
        value_type: "Int64",
        rows: 0,
        arguments: 1,
        concurrency: 1,
        partial_states: 0,
        state_bytes: 0,
        input_bytes: 0,
        output_bytes: 0,
        invocations: 0,
        errors: 0,
        samples: REPETITIONS,
        stats: lookup,
    });

    let input = int64_batch(4_096, 1);
    let function = runtime
        .find_function("add_one_int64", vec![DataType::Int64], DataType::Int64)
        .unwrap();
    let start = Instant::now();
    black_box(runtime.call(&function, &input).unwrap());
    let first = Stats::from_samples(&[start.elapsed().as_nanos()]);
    let start = Instant::now();
    black_box(runtime.call(&function, &input).unwrap());
    let second = Stats::from_samples(&[start.elapsed().as_nanos()]);
    let warm = measure(|| runtime.call(&function, &input).unwrap());
    for (path, stats, samples) in [
        ("first_invocation", first, 1),
        ("second_invocation", second, 1),
        ("steady_state", warm, REPETITIONS),
    ] {
        csv.row(Row {
            layer: "A",
            category: "scalar_cold_warm",
            path,
            value_type: "Int64",
            rows: input.num_rows(),
            arguments: 1,
            concurrency: 1,
            partial_states: 0,
            state_bytes: 0,
            input_bytes: input.get_array_memory_size(),
            output_bytes: 0,
            invocations: 1,
            errors: 0,
            samples,
            stats,
        });
    }
}

fn isolated_cold_warm(csv: &mut Csv, binary: &[u8]) {
    let compile = measure(|| runtime_with_pool(black_box(binary), Some(8)));
    csv.row(Row {
        layer: "A",
        category: "isolated_cold_warm",
        path: "module_compile_pool8",
        value_type: "Int64",
        rows: 0,
        arguments: 1,
        concurrency: 8,
        partial_states: 0,
        state_bytes: 0,
        input_bytes: binary.len(),
        output_bytes: 0,
        invocations: 0,
        errors: 0,
        samples: REPETITIONS,
        stats: compile,
    });

    let input = int64_batch(4_096, 1);
    for (label, pool_size) in [("on_demand", None), ("pool1", Some(1)), ("pool8", Some(8))] {
        let runtime = runtime_with_pool(binary, pool_size);
        let function = runtime
            .find_function("add_one_int64", vec![DataType::Int64], DataType::Int64)
            .unwrap();
        let start = Instant::now();
        black_box(runtime.isolated().call(&function, &input).unwrap());
        let first = Stats::from_samples(&[start.elapsed().as_nanos()]);
        let start = Instant::now();
        black_box(runtime.isolated().call(&function, &input).unwrap());
        let second = Stats::from_samples(&[start.elapsed().as_nanos()]);
        let warm = measure(|| runtime.isolated().call(&function, &input).unwrap());
        for (phase, stats, samples) in [
            ("first", first, 1),
            ("second", second, 1),
            ("warm", warm, REPETITIONS),
        ] {
            let path = format!("{label}_{phase}");
            csv.row(Row {
                layer: "A",
                category: "isolated_cold_warm",
                path: &path,
                value_type: "Int64",
                rows: input.num_rows(),
                arguments: 1,
                concurrency: pool_size.unwrap_or(1),
                partial_states: 0,
                state_bytes: 0,
                input_bytes: input.get_array_memory_size(),
                output_bytes: 0,
                invocations: 1,
                errors: 0,
                samples,
                stats,
            });
        }
    }
}

fn isolated_lifecycle(csv: &mut Csv, runtime: &Runtime) {
    let scalar_input = int64_batch(4_096, 1);
    let scalar = runtime
        .find_function("add_one_int64", vec![DataType::Int64], DataType::Int64)
        .unwrap();
    let scalar_output = runtime.isolated().call(&scalar, &scalar_input).unwrap();
    let scalar_stats = measure(|| {
        runtime
            .isolated()
            .call(&scalar, black_box(&scalar_input))
            .unwrap()
    });
    csv.row(Row {
        layer: "A",
        category: "isolated_scalar",
        path: "wasm_pool8",
        value_type: "Int64",
        rows: scalar_input.num_rows(),
        arguments: 1,
        concurrency: 1,
        partial_states: 0,
        state_bytes: 0,
        input_bytes: scalar_input.get_array_memory_size(),
        output_bytes: scalar_output.get_array_memory_size(),
        invocations: 1,
        errors: 0,
        samples: REPETITIONS,
        stats: scalar_stats,
    });

    let aggregate = runtime
        .find_aggregate_function("sum", vec![DataType::Float64], DataType::Float64)
        .unwrap();
    let initial = runtime
        .isolated()
        .create_aggregate_state(&aggregate)
        .unwrap();
    let input = float64_batch(4_096);
    let state = runtime
        .isolated()
        .update_aggregate_state(&aggregate, &initial, &input)
        .unwrap();
    let partials = Arc::new(BinaryArray::from_iter_values(
        (0..8).map(|_| state.as_slice()),
    )) as ArrayRef;
    let states = Arc::new(BinaryArray::from(vec![Some(state.as_slice())])) as ArrayRef;
    let cases = [
        (
            "wasm_init_pool8",
            0,
            0,
            measure(|| {
                runtime
                    .isolated()
                    .create_aggregate_state(&aggregate)
                    .unwrap()
            }),
        ),
        (
            "wasm_update_pool8",
            input.num_rows(),
            0,
            measure(|| {
                runtime
                    .isolated()
                    .update_aggregate_state(&aggregate, &initial, black_box(&input))
                    .unwrap()
            }),
        ),
        (
            "wasm_merge_pool8",
            0,
            8,
            measure(|| {
                runtime
                    .isolated()
                    .merge_aggregate_states(&aggregate, &initial, Arc::clone(&partials))
                    .unwrap()
            }),
        ),
        (
            "wasm_finish_pool8",
            1,
            1,
            measure(|| {
                runtime
                    .isolated()
                    .finish_aggregate_states(&aggregate, Arc::clone(&states))
                    .unwrap()
            }),
        ),
    ];
    for (path, rows, partial_states, stats) in cases {
        csv.row(Row {
            layer: "A",
            category: "isolated_aggregate_lifecycle",
            path,
            value_type: "Float64",
            rows,
            arguments: 1,
            concurrency: 1,
            partial_states,
            state_bytes: state.len(),
            input_bytes: input.get_array_memory_size(),
            output_bytes: state.len(),
            invocations: 1,
            errors: 0,
            samples: REPETITIONS,
            stats,
        });
    }
}

fn aggregate_matrices(csv: &mut Csv, runtime: &Runtime) {
    let sum = runtime
        .find_aggregate_function("sum", vec![DataType::Float64], DataType::Float64)
        .unwrap();
    let initial = runtime.create_aggregate_state(&sum).unwrap();
    assert_eq!(initial.len(), 10);
    let init = measure(|| runtime.create_aggregate_state(&sum).unwrap());
    csv.row(Row {
        layer: "A",
        category: "aggregate_lifecycle",
        path: "wasm_init",
        value_type: "Float64",
        rows: 0,
        arguments: 1,
        concurrency: 1,
        partial_states: 0,
        state_bytes: initial.len(),
        input_bytes: 0,
        output_bytes: initial.len(),
        invocations: 1,
        errors: 0,
        samples: REPETITIONS,
        stats: init,
    });

    for rows in BATCH_SIZES {
        let input = float64_batch(rows);
        let native = measure(|| {
            let values = input
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            values.iter().flatten().sum::<f64>()
        });
        let updated = runtime
            .update_aggregate_state(&sum, &initial, &input)
            .unwrap();
        let wasm = measure(|| {
            runtime
                .update_aggregate_state(&sum, &initial, black_box(&input))
                .unwrap()
        });
        for (path, stats) in [("native_sum", native), ("wasm_update", wasm)] {
            csv.row(Row {
                layer: "A",
                category: "aggregate_input",
                path,
                value_type: "Float64",
                rows,
                arguments: 1,
                concurrency: 1,
                partial_states: 0,
                state_bytes: updated.len(),
                input_bytes: input.get_array_memory_size(),
                output_bytes: updated.len(),
                invocations: usize::from(path == "wasm_update"),
                errors: 0,
                samples: REPETITIONS,
                stats,
            });
        }
    }

    let input = float64_batch(4_096);
    let state = runtime
        .update_aggregate_state(&sum, &initial, &input)
        .unwrap();
    let state_copy = measure(|| black_box(state.clone()));
    csv.row(Row {
        layer: "A",
        category: "aggregate_lifecycle",
        path: "host_state_clone",
        value_type: "Binary",
        rows: 0,
        arguments: 0,
        concurrency: 1,
        partial_states: 0,
        state_bytes: state.len(),
        input_bytes: state.len(),
        output_bytes: state.len(),
        invocations: 0,
        errors: 0,
        samples: REPETITIONS,
        stats: state_copy,
    });

    for partial_count in [1, 2, 4, 8, 16, 32] {
        let partials = Arc::new(BinaryArray::from_iter_values(
            (0..partial_count).map(|_| state.as_slice()),
        )) as ArrayRef;
        let merged = runtime
            .merge_aggregate_states(&sum, &initial, Arc::clone(&partials))
            .unwrap();
        let stats = measure(|| {
            runtime
                .merge_aggregate_states(&sum, &initial, Arc::clone(&partials))
                .unwrap()
        });
        csv.row(Row {
            layer: "A",
            category: "aggregate_merge",
            path: "wasm_merge",
            value_type: "Binary",
            rows: 0,
            arguments: 0,
            concurrency: 1,
            partial_states: partial_count,
            state_bytes: state.len(),
            input_bytes: partials.get_array_memory_size() + initial.len(),
            output_bytes: merged.len(),
            invocations: 1,
            errors: 0,
            samples: REPETITIONS,
            stats,
        });
    }

    let states = Arc::new(BinaryArray::from(vec![Some(state.as_slice())])) as ArrayRef;
    let finish = measure(|| {
        runtime
            .finish_aggregate_states(&sum, Arc::clone(&states))
            .unwrap()
    });
    csv.row(Row {
        layer: "A",
        category: "aggregate_lifecycle",
        path: "wasm_finish",
        value_type: "Float64",
        rows: 1,
        arguments: 0,
        concurrency: 1,
        partial_states: 1,
        state_bytes: state.len(),
        input_bytes: states.get_array_memory_size(),
        output_bytes: 8,
        invocations: 1,
        errors: 0,
        samples: REPETITIONS,
        stats: finish,
    });

    let passthrough = runtime
        .find_aggregate_function("bench_state", vec![DataType::Float64], DataType::Int64)
        .unwrap();
    let one = float64_batch(1);
    for state_bytes in [16, 64, 256, 1_024, 16_384, 65_536] {
        let state = vec![7_u8; state_bytes];
        let update = measure(|| {
            runtime
                .update_aggregate_state(&passthrough, &state, &one)
                .unwrap()
        });
        csv.row(Row {
            layer: "A",
            category: "aggregate_state_size",
            path: "wasm_update_passthrough",
            value_type: "Binary",
            rows: 1,
            arguments: 1,
            concurrency: 1,
            partial_states: 0,
            state_bytes,
            input_bytes: state_bytes + one.get_array_memory_size(),
            output_bytes: state_bytes,
            invocations: 1,
            errors: 0,
            samples: REPETITIONS,
            stats: update,
        });
        let partials = Arc::new(BinaryArray::from_iter_values(
            (0..8).map(|_| state.as_slice()),
        )) as ArrayRef;
        let merge = measure(|| {
            runtime
                .merge_aggregate_states(&passthrough, &state, Arc::clone(&partials))
                .unwrap()
        });
        csv.row(Row {
            layer: "A",
            category: "aggregate_state_size",
            path: "wasm_merge_passthrough",
            value_type: "Binary",
            rows: 0,
            arguments: 0,
            concurrency: 1,
            partial_states: 8,
            state_bytes,
            input_bytes: state_bytes + partials.get_array_memory_size(),
            output_bytes: state_bytes,
            invocations: 1,
            errors: 0,
            samples: REPETITIONS,
            stats: merge,
        });
    }
}

fn concurrent_scalar(csv: &mut Csv, runtime: Arc<Runtime>) {
    let input = Arc::new(int64_batch(4_096, 1));
    for concurrency in CONCURRENCY {
        let barrier = Arc::new(Barrier::new(concurrency + 1));
        let handles = (0..concurrency)
            .map(|_| {
                let runtime = Arc::clone(&runtime);
                let input = Arc::clone(&input);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let function = runtime
                        .find_function("add_one_int64", vec![DataType::Int64], DataType::Int64)
                        .unwrap();
                    for _ in 0..WARMUP {
                        black_box(runtime.call(&function, &input).unwrap());
                    }
                    barrier.wait();
                    let mut samples = Vec::with_capacity(REPETITIONS);
                    let mut errors = 0;
                    for _ in 0..REPETITIONS {
                        let started = Instant::now();
                        if runtime.call(&function, &input).is_err() {
                            errors += 1;
                        }
                        samples.push(started.elapsed().as_nanos());
                    }
                    (samples, errors)
                })
            })
            .collect::<Vec<_>>();
        let start = Instant::now();
        barrier.wait();
        let mut samples = Vec::with_capacity(concurrency * REPETITIONS);
        let mut errors = 0;
        for handle in handles {
            let (mut thread_samples, thread_errors) = handle.join().unwrap();
            samples.append(&mut thread_samples);
            errors += thread_errors;
        }
        let wall = start.elapsed();
        let stats = Stats::from_samples(&samples);
        let operations = concurrency * REPETITIONS;
        let throughput_stats = Stats {
            median_ns: wall.as_nanos() as f64 / operations as f64,
            mean_ns: wall.as_nanos() as f64 / operations as f64,
            p95_ns: stats.p95_ns,
            p99_ns: stats.p99_ns,
        };
        for (path, stats) in [
            ("wasm_latency", stats),
            ("wasm_throughput", throughput_stats),
        ] {
            csv.row(Row {
                layer: "A",
                category: "scalar_concurrency",
                path,
                value_type: "Int64",
                rows: input.num_rows(),
                arguments: 1,
                concurrency,
                partial_states: 0,
                state_bytes: 0,
                input_bytes: input.get_array_memory_size(),
                output_bytes: 0,
                invocations: operations,
                errors,
                samples: samples.len(),
                stats,
            });
        }
    }
}

fn concurrent_aggregate(csv: &mut Csv, runtime: Arc<Runtime>) {
    let input = Arc::new(float64_batch(4_096));
    for concurrency in CONCURRENCY {
        let barrier = Arc::new(Barrier::new(concurrency + 1));
        let handles = (0..concurrency)
            .map(|_| {
                let runtime = Arc::clone(&runtime);
                let input = Arc::clone(&input);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let aggregate = runtime
                        .find_aggregate_function("sum", vec![DataType::Float64], DataType::Float64)
                        .unwrap();
                    let initial = runtime.create_aggregate_state(&aggregate).unwrap();
                    for _ in 0..WARMUP {
                        black_box(
                            runtime
                                .update_aggregate_state(&aggregate, &initial, &input)
                                .unwrap(),
                        );
                    }
                    barrier.wait();
                    let mut samples = Vec::with_capacity(REPETITIONS);
                    let mut errors = 0;
                    for _ in 0..REPETITIONS {
                        let started = Instant::now();
                        if runtime
                            .update_aggregate_state(&aggregate, &initial, &input)
                            .is_err()
                        {
                            errors += 1;
                        }
                        samples.push(started.elapsed().as_nanos());
                    }
                    (samples, errors)
                })
            })
            .collect::<Vec<_>>();
        let start = Instant::now();
        barrier.wait();
        let mut samples = Vec::with_capacity(concurrency * REPETITIONS);
        let mut errors = 0;
        for handle in handles {
            let (mut thread_samples, thread_errors) = handle.join().unwrap();
            samples.append(&mut thread_samples);
            errors += thread_errors;
        }
        let wall = start.elapsed();
        let operations = concurrency * REPETITIONS;
        let stats = Stats::from_samples(&samples);
        let throughput_stats = Stats {
            median_ns: wall.as_nanos() as f64 / operations as f64,
            mean_ns: wall.as_nanos() as f64 / operations as f64,
            p95_ns: stats.p95_ns,
            p99_ns: stats.p99_ns,
        };
        for (path, stats) in [
            ("wasm_update_latency", stats),
            ("wasm_update_throughput", throughput_stats),
        ] {
            csv.row(Row {
                layer: "A",
                category: "aggregate_concurrency",
                path,
                value_type: "Float64",
                rows: input.num_rows(),
                arguments: 1,
                concurrency,
                partial_states: 0,
                state_bytes: 10,
                input_bytes: input.get_array_memory_size(),
                output_bytes: 10,
                invocations: operations,
                errors,
                samples: samples.len(),
                stats,
            });
        }
    }
}

fn profile(binary: &[u8], aggregate: bool) {
    let runtime = runtime(binary);
    let deadline = Instant::now() + Duration::from_secs(15);
    if aggregate {
        let function = runtime
            .find_aggregate_function("sum", vec![DataType::Float64], DataType::Float64)
            .unwrap();
        let initial = runtime.create_aggregate_state(&function).unwrap();
        let input = float64_batch(4_096);
        while Instant::now() < deadline {
            black_box(
                runtime
                    .update_aggregate_state(&function, &initial, &input)
                    .unwrap(),
            );
        }
    } else {
        let function = runtime
            .find_function("add_one_int64", vec![DataType::Int64], DataType::Int64)
            .unwrap();
        let input = int64_batch(4_096, 1);
        while Instant::now() < deadline {
            black_box(runtime.call(&function, &input).unwrap());
        }
    }
}

fn main() {
    let binary = fs::read(BINARY_PATH).expect("build arrow-udf-example for wasm32-wasip1 first");
    match std::env::var("WASM_BENCH_MODE").as_deref() {
        Ok("profile_scalar") => {
            profile(&binary, false);
            return;
        }
        Ok("profile_aggregate") => {
            profile(&binary, true);
            return;
        }
        Ok(mode) => panic!("unknown WASM_BENCH_MODE {mode}"),
        Err(_) => {}
    }

    let output = output_directory();
    fs::create_dir_all(&output).unwrap();
    let mut csv = Csv::create(&output.join("runtime.csv"));
    cold_warm(&mut csv, &binary);
    isolated_cold_warm(&mut csv, &binary);
    let runtime = Arc::new(runtime(&binary));
    scalar_matrices(&mut csv, &runtime);
    aggregate_matrices(&mut csv, &runtime);
    concurrent_scalar(&mut csv, Arc::clone(&runtime));
    concurrent_aggregate(&mut csv, runtime);
    let isolated_runtime = runtime_with_pool(&binary, Some(8));
    isolated_lifecycle(&mut csv, &isolated_runtime);
}
