# WebAssembly UDF for Apache Arrow

For untrusted user-defined functions, you can compile them into WebAssembly and run them in a sandboxed environment.

## Build UDF in WebAssembly

Create a project and add the following lines to your `Cargo.toml`:

```toml
[dependencies]
arrow-udf = "0.6"
```

Define your functions with the `#[function]` macro:

```rust,ignore
use arrow_udf::function;

#[function("gcd(int, int) -> int")]
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
```

Then compile the project into WebAssembly:

```sh
cargo build --release --target wasm32-wasip1
```

You can find the generated WebAssembly module in `target/wasm32-wasip1/release/*.wasm`.

## Run UDF in WebAssembly

Add the following lines to your `Cargo.toml`:

```toml
[dependencies]
arrow-udf-runtime = "0.7"
```

You can then load the WebAssembly module and call the functions:

```rust,ignore
use arrow_udf_runtime::wasm::Runtime;
use arrow_schema::DataType;
use arrow_array::RecordBatch;

// load the WebAssembly module
let binary = std::fs::read("udf.wasm").unwrap();
// create a runtime from the module
let runtime = Runtime::new(&binary).unwrap();
// list available functions in the module:
for name in runtime.functions() {
    println!("{}", name);
}
// call the function with a RecordBatch
let func = runtime.find_function("gcd", vec![DataType::Int32, DataType::Int32], DataType::Int32).unwrap();
let input: RecordBatch = ...;
let output = runtime.call(func, &input).unwrap();
```

The WebAssembly runtime is powered by [Wasmtime](https://wasmtime.dev/).
Notice that each WebAssembly instance can only run single-threaded, we maintain an instance pool internally to support parallel calls from multiple threads.

## Runtime controls

Use `Runtime::builder` for untrusted modules and set explicit limits for the
embedding application:

```rust,ignore
let runtime = Runtime::builder(&binary)
    .memory_size_limit(64 * 1024 * 1024)
    .table_elements_limit(10_000)
    .instance_limit(1)
    .table_limit(4)
    .memory_count_limit(4)
    .fuel_per_call(10_000_000)
    .epoch_deadline_ticks(2)
    .build()?;

let epoch = runtime.epoch_controller().unwrap();
// The embedding application advances `epoch` from its own timer or scheduler.
epoch.increment();
```

Memory and table-element limits apply to each memory or table in a Store.
Instance, table, and linear-memory counts also apply per Store. Fuel and the
relative epoch deadline are reset before every UDF invocation, including when
an instance is reused from the pool. Fuel exhaustion and epoch interruption
can be classified by downcasting the returned `anyhow::Error` to
`RuntimeControlError`.

The runtime never starts a timer thread. Epoch cadence and aggregate limits
across stores are responsibilities of the embedding application. Wasmtime 36
does not expose a per-Store compiled-module count limit, so a bounded module
cache must enforce that limit outside this runtime.

The default WASI Preview1 policy inherits no filesystem directories, sockets,
environment variables, process arguments, or stdin. It denies clock access and
uses the same deterministic random byte stream for every call. Polling and
sleeping are also denied, so synchronous host calls cannot bypass fuel and
epoch execution guards. Ambient clocks and randomness require an explicit
`WasiPolicy` opt-in. `Runtime::new` remains convenient and backward-compatible
for resource sizing, but production embedders should use the builder and set
finite resource and execution limits.

See the [example](./examples/wasm.rs) for more details. To run the example:

```sh
cargo build --release -p arrow-udf-example --target wasm32-wasip1
cargo run --example wasm -- target/wasm32-wasip1/release/arrow_udf_example.wasm
```

## Build WASM UDF at Runtime

With the `wasm-build` feature enabled (by default), you can build the WebAssembly module at runtime:

```rust,ignore
let manifest = r#"
[dependencies]
chrono = "0.4"
"#;

let script = r#"
use arrow_udf::function;

#[function("gcd(int, int) -> int")]
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
"#;
let binary = arrow_udf_runtime::wasm::build::build(manifest, script).unwrap();
```

See the [`build`] module for more details.
