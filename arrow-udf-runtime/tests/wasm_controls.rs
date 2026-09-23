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

#![cfg(feature = "wasm-runtime")]

use std::{sync::Arc, thread, time::Duration};

use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use arrow_udf_runtime::wasm::{
    Runtime, RuntimeBuilder, RuntimeControlError, WasiClockPolicy, WasiPolicy, WasiRandomPolicy,
};

fn isolated_builder(binary: &[u8], pool_size: usize) -> RuntimeBuilder<'_> {
    Runtime::builder(binary)
        .memory_size_limit(64 * 1024 * 1024)
        .table_elements_limit(100_000)
        .instance_limit(1)
        .table_limit(1)
        .memory_count_limit(1)
        .isolated_instance_pool_size(pool_size)
}

fn module(extra_imports: &str, extra_items: &str) -> Vec<u8> {
    wat::parse_str(format!(
        r#"
        (module
          {extra_imports}
          (memory (export "memory") 4)
          (global (export "ARROWUDF_VERSION_3_0") i32 (i32.const 0))
          (func (export "alloc") (param i32 i32) (result i32)
            i32.const 4096)
          (func (export "dealloc") (param i32 i32 i32))
          (func (export "record_batch_iterator_next") (param i32 i32))
          (func (export "record_batch_iterator_drop") (param i32))
          (func $return_input (param $input i32) (param $length i32) (param $output i32)
                (result i32)
            local.get $output
            local.get $input
            i32.store
            local.get $output
            i32.const 4
            i32.add
            local.get $length
            i32.store
            i32.const 0)
          {extra_items}
        )
        "#
    ))
    .expect("test WAT must compile")
}

fn compute_module() -> Vec<u8> {
    module(
        "",
        r#"
        (func (export "arrowudf_ZmluaXRlKGludDY0KS0$aW50NjQ")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          (local $i i32)
          (loop $work
            local.get $i
            i32.const 1
            i32.add
            local.tee $i
            i32.const 1000
            i32.lt_u
            br_if $work)
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        (func (export "arrowudf_Zm9yZXZlcihpbnQ2NCktPmludDY0")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          (loop $forever
            local.get $input
            drop
            br $forever)
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        (func (export "arrowudf_dHJhcChpbnQ2NCktPmludDY0")
              (param i32 i32 i32) (result i32)
          unreachable)
        (func (export "arrowudf_d3JvbmdfdHlwZShpbnQ2NCktPmZsb2F0NjQ")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        "#,
    )
}

fn memory_module() -> Vec<u8> {
    module(
        "",
        r#"
        (func (export "arrowudf_Z3Jvd193aXRoaW4oaW50NjQpLT5pbnQ2NA")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          i32.const 1
          memory.grow
          drop
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        (func (export "arrowudf_Z3Jvd19leGNlZWQoaW50NjQpLT5pbnQ2NA")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          i32.const 1024
          memory.grow
          drop
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        "#,
    )
}

fn table_module() -> Vec<u8> {
    module(
        "",
        r#"
        (table $table 1 funcref)
        (func (export "arrowudf_dGFibGVfZ3Jvd193aXRoaW4oaW50NjQpLT5pbnQ2NA")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          ref.null func
          i32.const 1
          table.grow $table
          drop
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        (func (export "arrowudf_dGFibGVfZ3Jvd19leGNlZWQoaW50NjQpLT5pbnQ2NA")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          ref.null func
          i32.const 100
          table.grow $table
          drop
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        "#,
    )
}

fn clock_module() -> Vec<u8> {
    module(
        r#"(import "wasi_snapshot_preview1" "clock_time_get"
                   (func $clock_time_get (param i32 i64 i32) (result i32)))"#,
        r#"
        (func (export "arrowudf_Y2xvY2tfZGVuaWVkKGludDY0KS0$aW50NjQ")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          i32.const 0
          i64.const 1
          i32.const 128
          call $clock_time_get
          i32.eqz
          if
            unreachable
          end
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        "#,
    )
}

fn random_module() -> Vec<u8> {
    module(
        r#"(import "wasi_snapshot_preview1" "random_get"
                   (func $random_get (param i32 i32) (result i32)))"#,
        r#"
        (func (export "arrowudf_cmFuZG9tX2RldGVybWluaXN0aWMoaW50NjQpLT5pbnQ2NA")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          i32.const 128
          i32.const 4
          call $random_get
          if
            unreachable
          end
          i32.const 128
          i32.load8_u
          i32.const 97
          i32.ne
          if
            unreachable
          end
          i32.const 129
          i32.load8_u
          i32.const 114
          i32.ne
          if
            unreachable
          end
          i32.const 130
          i32.load8_u
          i32.const 114
          i32.ne
          if
            unreachable
          end
          i32.const 131
          i32.load8_u
          i32.const 111
          i32.ne
          if
            unreachable
          end
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        "#,
    )
}

fn isolation_module() -> Vec<u8> {
    module(
        "",
        r#"
        (data (i32.const 128) "\07\00\00\00")
        (global $state (mut i32) (i32.const 0))
        (table $state_table 1 funcref)
        (func $marker)
        (elem declare func $marker)
        (func $mutate
          i32.const 128
          i32.const 99
          i32.store
          i32.const 1
          global.set $state
          i32.const 0
          ref.func $marker
          table.set $state_table
          i32.const 1
          memory.grow
          drop)
        (func $assert_pristine
          i32.const 128
          i32.load
          i32.const 7
          i32.eq
          if
          else
            unreachable
          end
          global.get $state
          i32.eqz
          if
          else
            unreachable
          end
          i32.const 0
          table.get $state_table
          ref.is_null
          if
          else
            unreachable
          end
          memory.size
          i32.const 4
          i32.eq
          if
          else
            unreachable
          end)
        (func (export "arrowudf_bXV0YXRlX3N0YXRlKGludDY0KS0$aW50NjQ")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          call $mutate
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        (func (export "arrowudf_YXNzZXJ0X3ByaXN0aW5lKGludDY0KS0$aW50NjQ")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          call $assert_pristine
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        (func (export "arrowudf_bXV0YXRlX2FuZF90cmFwKGludDY0KS0$aW50NjQ")
              (param i32 i32 i32) (result i32)
          call $mutate
          unreachable)
        (func (export "arrowudf_bXV0YXRlX3dyb25nX3R5cGUoaW50NjQpLT5mbG9hdDY0")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          call $mutate
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        (func (export "arrowudf_YXNzZXJ0X3RoZW5fbXV0YXRlKGludDY0KS0$aW50NjQ")
              (param $input i32) (param $length i32) (param $output i32) (result i32)
          call $assert_pristine
          call $mutate
          local.get $input
          local.get $length
          local.get $output
          call $return_input)
        "#,
    )
}

fn input() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(Int64Array::from(vec![Some(7), None]))],
    )
    .unwrap()
}

fn call_identity(runtime: &Runtime, name: &str) -> anyhow::Result<RecordBatch> {
    let function = runtime.find_function(name, vec![DataType::Int64], DataType::Int64)?;
    runtime.call(&function, &input())
}

fn call_isolated_identity(runtime: &Runtime, name: &str) -> anyhow::Result<RecordBatch> {
    let function = runtime.find_function(name, vec![DataType::Int64], DataType::Int64)?;
    runtime.isolated().call(&function, &input())
}

fn assert_identity(output: &RecordBatch) {
    let values = output
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(values.value(0), 7);
    assert!(values.is_null(1));
}

#[test]
fn fuel_is_reset_per_call_and_exhaustion_is_typed() {
    let binary = compute_module();
    let runtime = Runtime::builder(&binary)
        .fuel_per_call(50_000)
        .build()
        .unwrap();

    assert_identity(&call_identity(&runtime, "finite").unwrap());
    assert_identity(&call_identity(&runtime, "finite").unwrap());

    let error = call_identity(&runtime, "forever").unwrap_err();
    assert_eq!(
        error.downcast_ref::<RuntimeControlError>(),
        Some(&RuntimeControlError::FuelExhausted)
    );

    assert_identity(&call_identity(&runtime, "finite").unwrap());
}

#[test]
fn epoch_interrupts_one_call_and_next_call_recovers() {
    let binary = compute_module();
    let runtime = Arc::new(
        Runtime::builder(&binary)
            .epoch_deadline_ticks(1)
            .build()
            .unwrap(),
    );
    let controller = runtime.epoch_controller().unwrap();
    let worker_runtime = Arc::clone(&runtime);
    let worker = thread::spawn(move || call_identity(&worker_runtime, "forever"));

    thread::sleep(Duration::from_millis(10));
    for _ in 0..100 {
        assert!(controller.increment());
        if worker.is_finished() {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }

    let error = worker
        .join()
        .expect("epoch worker must not panic")
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<RuntimeControlError>(),
        Some(&RuntimeControlError::EpochDeadlineExceeded)
    );
    assert_identity(&call_identity(&runtime, "finite").unwrap());
}

#[test]
fn fuel_and_epoch_are_independent_across_concurrent_calls() {
    let binary = compute_module();
    let runtime = Arc::new(
        Runtime::builder(&binary)
            .fuel_per_call(50_000)
            .epoch_deadline_ticks(100)
            .build()
            .unwrap(),
    );

    let workers = (0..8)
        .map(|_| {
            let runtime = Arc::clone(&runtime);
            thread::spawn(move || call_identity(&runtime, "finite"))
        })
        .collect::<Vec<_>>();
    for worker in workers {
        assert_identity(&worker.join().expect("worker must not panic").unwrap());
    }
}

#[test]
fn memory_growth_is_bounded_and_host_recovers() {
    let binary = memory_module();
    let runtime = Runtime::builder(&binary)
        .memory_size_limit(6 * 65_536)
        .build()
        .unwrap();

    assert_identity(&call_identity(&runtime, "grow_within").unwrap());
    assert!(call_identity(&runtime, "grow_exceed").is_err());
    assert_identity(&call_identity(&runtime, "grow_within").unwrap());
}

#[test]
fn table_growth_is_bounded_and_host_recovers() {
    let binary = table_module();
    let runtime = Runtime::builder(&binary)
        .table_elements_limit(3)
        .build()
        .unwrap();

    assert_identity(&call_identity(&runtime, "table_grow_within").unwrap());
    assert!(call_identity(&runtime, "table_grow_exceed").is_err());
    assert_identity(&call_identity(&runtime, "table_grow_within").unwrap());
}

#[test]
fn store_object_counts_are_enforced() {
    let compute = compute_module();
    let no_instances = Runtime::builder(&compute)
        .instance_limit(0)
        .build()
        .unwrap();
    assert!(call_identity(&no_instances, "finite").is_err());

    let no_memories = Runtime::builder(&compute)
        .memory_count_limit(0)
        .build()
        .unwrap();
    assert!(call_identity(&no_memories, "finite").is_err());

    let table = table_module();
    let no_tables = Runtime::builder(&table).table_limit(0).build().unwrap();
    assert!(call_identity(&no_tables, "table_grow_within").is_err());
}

#[test]
fn clocks_are_denied_by_default() {
    let binary = clock_module();
    let runtime = Runtime::new(&binary).unwrap();
    assert_identity(&call_identity(&runtime, "clock_denied").unwrap());
}

#[test]
fn randomness_is_deterministic_by_default() {
    let binary = random_module();
    let runtime = Runtime::new(&binary).unwrap();
    assert_identity(&call_identity(&runtime, "random_deterministic").unwrap());
    assert_identity(&call_identity(&runtime, "random_deterministic").unwrap());
}

#[test]
fn ambient_clock_and_random_are_explicit_opt_ins() {
    let policy = WasiPolicy {
        clocks: WasiClockPolicy::Ambient,
        random: WasiRandomPolicy::Ambient,
    };
    let binary = compute_module();
    let runtime = Runtime::builder(&binary)
        .wasi_policy(policy)
        .build()
        .unwrap();
    assert_identity(&call_identity(&runtime, "finite").unwrap());
}

#[test]
fn ordinary_guest_trap_does_not_panic_or_poison_runtime() {
    let binary = compute_module();
    let runtime = Runtime::new(&binary).unwrap();
    let error = call_identity(&runtime, "trap").unwrap_err();
    assert!(error.downcast_ref::<RuntimeControlError>().is_none());
    assert_identity(&call_identity(&runtime, "finite").unwrap());
}

#[test]
fn scalar_output_type_mismatch_is_reported() {
    let binary = compute_module();
    let runtime = Runtime::new(&binary).unwrap();
    let function = runtime
        .find_function("wrong_type", vec![DataType::Int64], DataType::Float64)
        .unwrap();
    let error = runtime.call(&function, &input()).unwrap_err();
    assert!(error.to_string().contains("output type mismatch"));
}

#[test]
fn isolated_calls_restore_memory_globals_tables_and_memory_size() {
    let binary = isolation_module();
    let runtime = isolated_builder(&binary, 2)
        .memory_size_limit(6 * 65_536)
        .table_elements_limit(2)
        .build()
        .unwrap();

    assert_identity(&call_isolated_identity(&runtime, "assert_pristine").unwrap());
    assert_identity(&call_isolated_identity(&runtime, "mutate_state").unwrap());
    assert_identity(&call_isolated_identity(&runtime, "assert_pristine").unwrap());
}

#[test]
fn isolated_calls_discard_state_after_trap_and_output_validation_failure() {
    let binary = isolation_module();
    let runtime = isolated_builder(&binary, 1)
        .memory_size_limit(6 * 65_536)
        .table_elements_limit(2)
        .build()
        .unwrap();

    assert!(call_isolated_identity(&runtime, "mutate_and_trap").is_err());
    assert_identity(&call_isolated_identity(&runtime, "assert_pristine").unwrap());

    let wrong_type = runtime
        .find_function(
            "mutate_wrong_type",
            vec![DataType::Int64],
            DataType::Float64,
        )
        .unwrap();
    let error = runtime.isolated().call(&wrong_type, &input()).unwrap_err();
    assert!(error.to_string().contains("output type mismatch"));
    assert_identity(&call_isolated_identity(&runtime, "assert_pristine").unwrap());
}

#[test]
fn isolated_calls_reset_fuel_after_failure() {
    let binary = compute_module();
    let runtime = isolated_builder(&binary, 1)
        .fuel_per_call(50_000)
        .build()
        .unwrap();

    let forever = runtime
        .find_function("forever", vec![DataType::Int64], DataType::Int64)
        .unwrap();
    let error = runtime.isolated().call(&forever, &input()).unwrap_err();
    assert_eq!(
        error.downcast_ref::<RuntimeControlError>(),
        Some(&RuntimeControlError::FuelExhausted)
    );
    assert_identity(&call_isolated_identity(&runtime, "finite").unwrap());
}

#[test]
fn isolated_calls_reset_deadline_after_timeout() {
    let binary = compute_module();
    let runtime = Arc::new(
        isolated_builder(&binary, 1)
            .epoch_deadline_ticks(1)
            .build()
            .unwrap(),
    );
    let controller = runtime.epoch_controller().unwrap();
    let worker_runtime = Arc::clone(&runtime);
    let worker = thread::spawn(move || {
        let forever =
            worker_runtime.find_function("forever", vec![DataType::Int64], DataType::Int64)?;
        worker_runtime.isolated().call(&forever, &input())
    });

    thread::sleep(Duration::from_millis(10));
    for _ in 0..100 {
        assert!(controller.increment());
        if worker.is_finished() {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }

    let error = worker
        .join()
        .expect("epoch worker must not panic")
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<RuntimeControlError>(),
        Some(&RuntimeControlError::EpochDeadlineExceeded)
    );
    assert_identity(&call_isolated_identity(&runtime, "finite").unwrap());
}

#[test]
fn isolated_calls_are_pristine_across_functions_and_concurrent_callers() {
    let binary = isolation_module();
    let runtime = Arc::new(
        isolated_builder(&binary, 8)
            .memory_size_limit(6 * 65_536)
            .table_elements_limit(2)
            .build()
            .unwrap(),
    );

    assert_identity(&call_isolated_identity(&runtime, "mutate_state").unwrap());
    assert_identity(&call_isolated_identity(&runtime, "assert_pristine").unwrap());

    let workers = (0..8)
        .map(|_| {
            let runtime = Arc::clone(&runtime);
            thread::spawn(move || {
                for _ in 0..16 {
                    assert_identity(
                        &call_isolated_identity(&runtime, "assert_then_mutate").unwrap(),
                    );
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().expect("isolation worker must not panic");
    }
}

#[test]
fn isolated_wasi_random_state_is_fresh_per_call() {
    let binary = random_module();
    let runtime = isolated_builder(&binary, 1).build().unwrap();

    assert_identity(&call_isolated_identity(&runtime, "random_deterministic").unwrap());
    assert_identity(&call_isolated_identity(&runtime, "random_deterministic").unwrap());
}

#[test]
fn isolated_pool_preserves_memory_and_table_limits() {
    let memory = memory_module();
    let memory_runtime = isolated_builder(&memory, 1)
        .memory_size_limit(6 * 65_536)
        .build()
        .unwrap();
    assert_identity(&call_isolated_identity(&memory_runtime, "grow_within").unwrap());
    assert!(call_isolated_identity(&memory_runtime, "grow_exceed").is_err());
    assert_identity(&call_isolated_identity(&memory_runtime, "grow_within").unwrap());

    let table = table_module();
    let table_runtime = isolated_builder(&table, 1)
        .table_elements_limit(3)
        .build()
        .unwrap();
    assert_identity(&call_isolated_identity(&table_runtime, "table_grow_within").unwrap());
    assert!(call_isolated_identity(&table_runtime, "table_grow_exceed").is_err());
    assert_identity(&call_isolated_identity(&table_runtime, "table_grow_within").unwrap());
}

#[test]
fn isolated_instance_pool_size_must_be_nonzero() {
    let binary = compute_module();
    let error = isolated_builder(&binary, 0).build().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("isolated_instance_pool_size must be greater than zero")
    );
}

#[test]
fn isolated_instance_pool_requires_explicit_resource_bounds() {
    let binary = compute_module();
    let error = Runtime::builder(&binary)
        .isolated_instance_pool_size(1)
        .build()
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("isolated instance pooling requires memory_size_limit")
    );
}
