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

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, RecordBatchOptions,
    StringArray, UInt64Array,
};
use arrow_cast::pretty::pretty_format_batches;
use arrow_schema::{DataType, Field, Fields, Schema};
use arrow_udf_runtime::wasm::Runtime;
use expect_test::{Expect, expect};

const BINARY_PATH: &str = "../target/wasm32-wasip1/release/arrow_udf_example.wasm";

static RUNTIME: std::sync::LazyLock<Runtime> = std::sync::LazyLock::new(|| {
    Runtime::new(&std::fs::read(BINARY_PATH).expect("failed to load wasm binary"))
        .expect("failed to create wasm runtime")
});

#[test]
fn test_oom() {
    let input = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::default().with_row_count(Some(1)),
    )
    .unwrap();

    let oom = RUNTIME
        .find_function("oom", Vec::<Field>::new(), DataType::Null)
        .unwrap();
    let output = RUNTIME.call(&oom, &input);
    // panic message should be contained in the error message
    assert!(
        output
            .unwrap_err()
            .to_string()
            .contains("capacity overflow")
    );
}

#[test]
fn test_sleep() {
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();

    let sleep = RUNTIME
        .find_function("sleep", vec![DataType::Int32], DataType::Int32)
        .unwrap();
    let output = RUNTIME.call(&sleep, &input);
    output.unwrap_err();
}

#[test]
fn test_gcd() {
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![Some(15), Some(5), None])),
            Arc::new(Int32Array::from(vec![25, 0, 1])),
        ],
    )
    .unwrap();

    let gcd = RUNTIME
        .find_function(
            "gcd",
            vec![DataType::Int32, DataType::Int32],
            DataType::Int32,
        )
        .unwrap();
    let output = RUNTIME.call(&gcd, &input).unwrap();
    check(
        &[output],
        expect![[r#"
        +-----+
        | gcd |
        +-----+
        | 5   |
        | 5   |
        |     |
        +-----+"#]],
    );
}

#[test]
fn test_division_by_zero() {
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![Some(15), Some(5), None])),
            Arc::new(Int32Array::from(vec![25, 0, 1])),
        ],
    )
    .unwrap();

    let div = RUNTIME
        .find_function(
            "div",
            vec![DataType::Int32, DataType::Int32],
            DataType::Int32,
        )
        .unwrap();
    let output = RUNTIME.call(&div, &input).unwrap();
    check(
        &[output],
        expect![[r#"
        +-----+------------------+
        | div | error            |
        +-----+------------------+
        | 0   |                  |
        |     | division by zero |
        |     |                  |
        +-----+------------------+"#]],
    );
}

#[test]
fn test_length() {
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
        vec![Arc::new(StringArray::from(vec!["rising", "wave"]))],
    )
    .unwrap();

    let length = RUNTIME
        .find_function("length", vec![DataType::Utf8], DataType::Int32)
        .unwrap();
    let output = RUNTIME.call(&length, &input).unwrap();
    check(
        &[output],
        expect![[r#"
        +--------+
        | length |
        +--------+
        | 6      |
        | 4      |
        +--------+"#]],
    );
}

#[test]
fn test_scalar_v1_types_and_nulls() {
    let cases: Vec<(DataType, Arc<dyn Array>, &str)> = vec![
        (
            DataType::Boolean,
            Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
            "boolean_not",
        ),
        (
            DataType::Int64,
            Arc::new(Int64Array::from(vec![Some(1), None, Some(-2)])),
            "add_one_int64",
        ),
        (
            DataType::UInt64,
            Arc::new(UInt64Array::from(vec![Some(1), None, Some(2)])),
            "add_one_uint64",
        ),
        (
            DataType::Float64,
            Arc::new(Float64Array::from(vec![Some(1.5), None, Some(-2.25)])),
            "double_float64",
        ),
        (
            DataType::Utf8,
            Arc::new(StringArray::from(vec![Some("abc"), Some("时序"), None])),
            "append_wasm",
        ),
    ];

    for (data_type, values, function_name) in cases {
        let input = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                data_type.clone(),
                true,
            )])),
            vec![values],
        )
        .unwrap();
        let function = RUNTIME
            .find_function(function_name, vec![data_type.clone()], data_type)
            .unwrap();
        let output = RUNTIME.call(&function, &input).unwrap();
        assert_eq!(output.num_rows(), input.num_rows());
        assert_eq!(output.column(0).null_count(), input.column(0).null_count());
        match function_name {
            "boolean_not" => {
                let values = output
                    .column(0)
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap();
                assert!(!values.value(0));
                assert!(values.is_null(1));
                assert!(values.value(2));
            }
            "add_one_int64" => {
                let values = output
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                assert_eq!(values.value(0), 2);
                assert!(values.is_null(1));
                assert_eq!(values.value(2), -1);
            }
            "add_one_uint64" => {
                let values = output
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                assert_eq!(values.value(0), 2);
                assert!(values.is_null(1));
                assert_eq!(values.value(2), 3);
            }
            "double_float64" => {
                let values = output
                    .column(0)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                assert_eq!(values.value(0), 3.0);
                assert!(values.is_null(1));
                assert_eq!(values.value(2), -4.5);
            }
            "append_wasm" => {
                let values = output
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                assert_eq!(values.value(0), "abc-wasm");
                assert_eq!(values.value(1), "时序-wasm");
                assert!(values.is_null(2));
            }
            _ => unreachable!("unexpected test function"),
        }
    }
}

#[test]
fn test_default_sandbox_denies_filesystem_network_and_environment() {
    let input = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::default().with_row_count(Some(1)),
    )
    .unwrap();

    let create_file = RUNTIME
        .find_function("create_file", Vec::<DataType>::new(), DataType::Null)
        .unwrap();
    assert!(RUNTIME.call(&create_file, &input).is_err());

    for name in ["environment_access", "network_access"] {
        let function = RUNTIME
            .find_function(name, Vec::<DataType>::new(), DataType::Int64)
            .unwrap();
        let output = RUNTIME.call(&function, &input).unwrap();
        let values = output
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.value(0), 0, "{name} unexpectedly succeeded");
    }
}

#[test]
fn test_real_guest_memory_limit_and_recovery() {
    let binary = std::fs::read(BINARY_PATH).unwrap();
    let runtime = Runtime::builder(&binary)
        .memory_size_limit(8 * 1024 * 1024)
        .build()
        .unwrap();
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "bytes",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![32 * 1024 * 1024]))],
    )
    .unwrap();
    let allocate = runtime
        .find_function("allocate_bytes", vec![DataType::Int64], DataType::Int64)
        .unwrap();
    assert!(runtime.call(&allocate, &input).is_err());

    let healthy = runtime
        .find_function("add_one_int64", vec![DataType::Int64], DataType::Int64)
        .unwrap();
    let healthy_input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![1]))],
    )
    .unwrap();
    assert!(runtime.call(&healthy, &healthy_input).is_ok());
}

#[test]
fn test_concurrent_callers_have_independent_instances() {
    let workers = (0_i64..8)
        .map(|value| {
            std::thread::spawn(move || {
                let input = RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new(
                        "value",
                        DataType::Int64,
                        false,
                    )])),
                    vec![Arc::new(Int64Array::from(vec![value]))],
                )
                .unwrap();
                let function = RUNTIME
                    .find_function("add_one_int64", vec![DataType::Int64], DataType::Int64)
                    .unwrap();
                let output = RUNTIME.call(&function, &input).unwrap();
                let values = output
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                assert_eq!(values.value(0), value + 1);
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().expect("concurrent caller must not panic");
    }
}

#[test]
fn test_existing_abi3_guest_when_provided() {
    let Some(path) = std::env::var_os("ARROW_UDF_ABI3_GUEST") else {
        return;
    };
    let runtime = Runtime::new(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(runtime.abi_version(), (3, 0));
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )])),
        vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]))],
    )
    .unwrap();
    let function = runtime
        .find_function("add_one", vec![DataType::Int64], DataType::Int64)
        .unwrap();
    let output = runtime.call(&function, &input).unwrap();
    let values = output
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(values.value(0), 2);
    assert!(values.is_null(1));
    assert_eq!(values.value(2), 4);
}

#[test]
fn test_key_value() {
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
        vec![Arc::new(StringArray::from(vec!["rising=wave", "???"]))],
    )
    .unwrap();

    let key_value = RUNTIME
        .find_function(
            "key_value",
            vec![DataType::Utf8],
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, true),
                Field::new("value", DataType::Utf8, true),
            ])),
        )
        .unwrap();
    let output = RUNTIME.call(&key_value, &input).unwrap();
    check(
        &[output],
        expect![[r#"
        +----------------------------+
        | key_value                  |
        +----------------------------+
        | {key: rising, value: wave} |
        |                            |
        +----------------------------+"#]],
    );
}

#[test]
fn test_range() {
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))],
    )
    .unwrap();

    let range = RUNTIME
        .find_table_function("range", vec![DataType::Int32], DataType::Int32)
        .unwrap();
    let mut iter = RUNTIME.call_table_function(&range, &input).unwrap();
    let output = iter.next().unwrap().unwrap();
    check(
        &[output],
        expect![[r#"
        +-----+-------+
        | row | range |
        +-----+-------+
        | 0   | 0     |
        | 2   | 0     |
        | 2   | 1     |
        | 2   | 2     |
        +-----+-------+"#]],
    );
}

/// Compare the actual output with the expected output.
#[track_caller]
fn check(actual: &[RecordBatch], expect: Expect) {
    expect.assert_eq(&pretty_format_batches(actual).unwrap().to_string());
}
