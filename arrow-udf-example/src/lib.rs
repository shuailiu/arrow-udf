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

use arrow_udf::{function, types::*};
use std::fmt::Write;

use arrow_udf::codegen::arrow_array::{Array, BinaryArray, Float64Array, ListArray};

#[function("gcd(int, int) -> int")]
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[function("div(int, int) -> int")]
fn div(x: i32, y: i32) -> Result<i32, &'static str> {
    if y == 0 {
        return Err("division by zero");
    }
    Ok(x / y)
}

#[function("segfault()")]
fn segfault() {
    unsafe { (usize::MAX as *const i32).read_volatile() };
}

#[function("oom()")]
fn oom() {
    _ = vec![0u8; usize::MAX];
}

#[function("create_file()")]
fn create_file() {
    std::fs::File::create("test").unwrap();
}

#[function("sleep(int) -> int")]
fn sleep(second: i32) -> i32 {
    std::thread::sleep(std::time::Duration::from_secs(second as u64));
    0
}

#[function("boolean_not(boolean) -> boolean")]
fn boolean_not(value: bool) -> bool {
    !value
}

#[function("add_one_int64(int64) -> int64")]
fn add_one_int64(value: i64) -> i64 {
    value + 1
}

#[function("add_one_uint64(uint64) -> uint64")]
fn add_one_uint64(value: u64) -> u64 {
    value + 1
}

#[function("double_float64(float64) -> float64")]
fn double_float64(value: f64) -> f64 {
    value * 2.0
}

#[function("append_wasm(string) -> string")]
fn append_wasm(value: &str, output: &mut impl Write) {
    write!(output, "{value}-wasm").unwrap();
}

#[function("allocate_bytes(int64) -> int64")]
fn allocate_bytes(bytes: i64) -> i64 {
    let allocation = vec![7_u8; bytes as usize];
    std::hint::black_box(&allocation);
    allocation.len() as i64
}

#[function("environment_access() -> int64")]
fn environment_access() -> i64 {
    std::env::vars().next().is_some() as i64
}

#[function("network_access() -> int64")]
fn network_access() -> i64 {
    std::net::TcpStream::connect("127.0.0.1:9").is_ok() as i64
}

#[function("decimal_add(decimal, decimal) -> decimal")]
fn decimal_add(a: Decimal, b: Decimal) -> Decimal {
    a + b
}

#[function("datetime(date, time) -> timestamp")]
fn datetime(date: NaiveDate, time: NaiveTime) -> NaiveDateTime {
    NaiveDateTime::new(date, time)
}

#[function("jsonb_access(json, int) -> json")]
fn jsonb_access(json: serde_json::Value, index: i32) -> Option<serde_json::Value> {
    json.get(index as usize).cloned()
}

#[function("length(varchar) -> int")]
#[function("length(bytea) -> int")]
fn length(s: impl AsRef<[u8]>) -> i32 {
    s.as_ref().len() as i32
}

#[derive(StructType)]
struct KeyValue<'a> {
    key: &'a str,
    value: &'a str,
}

#[function("key_value(varchar) -> struct KeyValue")]
fn key_value(kv: &str) -> Option<KeyValue<'_>> {
    let (key, value) = kv.split_once('=')?;
    Some(KeyValue { key, value })
}

#[function("range(int) -> setof int")]
fn range(x: i32) -> impl Iterator<Item = i32> {
    0..x
}

const SUM_STATE_VERSION: u8 = 1;

fn encode_sum_state(sum: f64, seen: bool) -> Vec<u8> {
    let mut state = Vec::with_capacity(10);
    state.push(SUM_STATE_VERSION);
    state.push(u8::from(seen));
    state.extend_from_slice(&sum.to_le_bytes());
    state
}

fn decode_sum_state(state: &[u8]) -> (f64, bool) {
    assert_eq!(state.len(), 10, "invalid sum aggregate state length");
    assert_eq!(
        state[0], SUM_STATE_VERSION,
        "invalid sum aggregate state version"
    );
    let sum = f64::from_le_bytes(state[2..].try_into().unwrap());
    (sum, state[1] != 0)
}

#[function("sum_init() -> binary")]
fn sum_init() -> Vec<u8> {
    encode_sum_state(0.0, false)
}

#[function(
    "sum_update(binary, float64[]) -> binary",
    batch_fn = "sum_update_batch"
)]
fn sum_update(_state: &[u8], _values: &[f64]) -> Vec<u8> {
    unreachable!("aggregate update uses its vectorized batch function")
}

fn sum_update_batch(states: &BinaryArray, values: &ListArray) -> BinaryArray {
    let (mut sum, mut seen) = decode_sum_state(states.value(0));
    let values = values.value(0);
    let values = values.as_any().downcast_ref::<Float64Array>().unwrap();
    for value in values.iter().flatten() {
        sum += value;
        seen = true;
    }
    let state = encode_sum_state(sum, seen);
    BinaryArray::from(vec![Some(state.as_slice())])
}

#[function("sum_merge(binary, binary[]) -> binary", batch_fn = "sum_merge_batch")]
fn sum_merge(_state: &[u8], _partial_states: &BinaryArray) -> Vec<u8> {
    unreachable!("aggregate merge uses its vectorized batch function")
}

fn sum_merge_batch(states: &BinaryArray, partial_states: &ListArray) -> BinaryArray {
    let (mut sum, mut seen) = decode_sum_state(states.value(0));
    let partial_states = partial_states.value(0);
    let partial_states = partial_states
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    for partial in partial_states.iter().flatten() {
        let (partial_sum, partial_seen) = decode_sum_state(partial);
        sum += partial_sum;
        seen |= partial_seen;
    }
    let state = encode_sum_state(sum, seen);
    BinaryArray::from(vec![Some(state.as_slice())])
}

#[function("sum_finish(binary) -> float64")]
fn sum_finish(state: &[u8]) -> Option<f64> {
    let (sum, seen) = decode_sum_state(state);
    seen.then_some(sum)
}
