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

#![doc = include_str!("README.md")]

use anyhow::{Context, Error, anyhow, bail, ensure};
use arrow_array::{Array, ArrayRef, BinaryArray, ListArray, RecordBatch, RecordBatchOptions};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, IntervalUnit, Schema, TimeUnit};
use itertools::Itertools;
use ram_file::{RamFile, RamFileRef};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use wasi_common::{
    Error as WasiError, ErrorExt, Poll, WasiClocks, WasiCtx, WasiSched,
    random::Deterministic,
    sched::Duration as WasiDuration,
    sync::{clocks_ctx, random_ctx},
    table::Table,
};
use wasmtime::*;

use crate::into_field::IntoField;

#[cfg(feature = "wasm-build")]
pub mod build;
mod ram_file;

/// The WASM UDF runtime.
///
/// This runtime contains an instance pool and can be shared by multiple threads.
pub struct Runtime {
    module: Module,
    /// Configurations.
    config: Config,
    /// Export names of user-defined functions.
    functions: Arc<HashSet<String>>,
    /// User-defined types.
    types: Arc<HashMap<String, String>>,
    /// Instance pool.
    instances: Mutex<Vec<Instance>>,
    /// ABI version. (major, minor)
    abi_version: (u8, u8),
}

/// Configurations for a WASM UDF runtime.
///
/// Store limits are enforced independently for every pooled instance. Fuel
/// and epoch deadlines are reset to their configured values before every UDF
/// invocation. Compiled-module count and aggregate limits across multiple
/// stores belong to the embedding application's module cache and scheduler.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[non_exhaustive]
pub struct Config {
    /// Maximum bytes for each guest linear memory.
    ///
    /// `None` preserves the legacy unbounded-memory behavior. Production
    /// embedders should always set an explicit value.
    pub memory_size_limit: Option<usize>,
    /// Maximum number of elements in each guest table.
    pub table_elements_limit: Option<usize>,
    /// Maximum number of instances in each Wasmtime Store.
    pub instance_limit: Option<usize>,
    /// Maximum number of tables in each Wasmtime Store.
    pub table_limit: Option<usize>,
    /// Maximum number of linear memories in each Wasmtime Store.
    pub memory_count_limit: Option<usize>,
    /// Fuel made available at the start of every UDF invocation.
    ///
    /// `None` disables fuel instrumentation. A configured value must be
    /// greater than zero.
    pub fuel_per_call: Option<u64>,
    /// Epoch ticks allowed from the current engine epoch for every invocation.
    ///
    /// The embedding application advances epochs through [`EpochController`].
    /// `None` disables epoch instrumentation. A configured value must be
    /// greater than zero.
    pub epoch_deadline_ticks: Option<u64>,
    /// WASI capabilities visible to guest modules.
    pub wasi: WasiPolicy,
    /// Maximum captured stdout/stderr size in bytes, per stream and instance.
    pub file_size_limit: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            memory_size_limit: None,
            table_elements_limit: None,
            instance_limit: None,
            table_limit: None,
            memory_count_limit: None,
            fuel_per_call: None,
            epoch_deadline_ticks: None,
            wasi: WasiPolicy::default(),
            file_size_limit: None,
        }
    }
}

/// Clock capabilities exposed through WASI Preview1.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum WasiClockPolicy {
    /// Return a WASI error for realtime and monotonic clock requests.
    #[default]
    Denied,
    /// Expose host realtime and monotonic clocks.
    Ambient,
}

/// Randomness exposed through WASI Preview1.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum WasiRandomPolicy {
    /// Return the same deterministic byte stream in every invocation.
    #[default]
    Deterministic,
    /// Expose nondeterministic host randomness.
    Ambient,
}

/// WASI Preview1 capability policy.
///
/// Filesystem directories, sockets, environment variables, process arguments,
/// and stdin are never inherited. Clocks and randomness are the only ambient
/// capabilities that can be enabled explicitly.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct WasiPolicy {
    /// Guest clock policy.
    pub clocks: WasiClockPolicy,
    /// Guest randomness policy.
    pub random: WasiRandomPolicy,
}

struct DeniedWasiScheduler;

#[async_trait::async_trait]
impl WasiSched for DeniedWasiScheduler {
    async fn poll_oneoff<'a>(&self, _poll: &mut Poll<'a>) -> std::result::Result<(), WasiError> {
        Err(WasiError::badf().context("WASI polling is not supported"))
    }

    async fn sched_yield(&self) -> std::result::Result<(), WasiError> {
        Ok(())
    }

    async fn sleep(&self, _duration: WasiDuration) -> std::result::Result<(), WasiError> {
        Err(WasiError::badf().context("WASI sleeping is not supported"))
    }
}

/// Typed execution-control failures returned through `anyhow::Error`.
///
/// Callers can classify these errors with `Error::downcast_ref` while the
/// existing public `anyhow::Result` API remains source-compatible.
#[derive(Debug, thiserror::Error, PartialEq, Eq, Clone, Copy)]
pub enum RuntimeControlError {
    /// The configured per-call fuel budget was exhausted.
    #[error("WASM UDF exhausted its per-call fuel budget")]
    FuelExhausted,
    /// The configured per-call epoch deadline was reached.
    #[error("WASM UDF exceeded its epoch deadline")]
    EpochDeadlineExceeded,
}

/// A weak handle used by an embedding application to advance engine epochs.
///
/// arrow-udf does not start a hidden timer thread. An application that uses
/// epoch deadlines should advance this controller from its own ticker. The
/// weak handle does not keep the runtime or engine alive.
#[derive(Clone)]
pub struct EpochController {
    engine: EngineWeak,
}

impl EpochController {
    /// Increment the engine epoch, returning `false` if the runtime was dropped.
    pub fn increment(&self) -> bool {
        let Some(engine) = self.engine.upgrade() else {
            return false;
        };
        #[cfg(target_has_atomic = "64")]
        engine.increment_epoch();
        #[cfg(not(target_has_atomic = "64"))]
        let _ = engine;
        cfg!(target_has_atomic = "64")
    }
}

/// Builder for a [`Runtime`] and its Wasmtime engine.
pub struct RuntimeBuilder<'a> {
    binary: &'a [u8],
    config: Config,
}

impl<'a> RuntimeBuilder<'a> {
    /// Create a builder for a WASM binary.
    pub fn new(binary: &'a [u8]) -> Self {
        Self {
            binary,
            config: Config::default(),
        }
    }

    /// Replace the complete runtime policy.
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Set the maximum bytes for each guest linear memory.
    pub fn memory_size_limit(mut self, limit: usize) -> Self {
        self.config.memory_size_limit = Some(limit);
        self
    }

    /// Set the maximum elements for each guest table.
    pub fn table_elements_limit(mut self, limit: usize) -> Self {
        self.config.table_elements_limit = Some(limit);
        self
    }

    /// Set the maximum instances in each Store.
    pub fn instance_limit(mut self, limit: usize) -> Self {
        self.config.instance_limit = Some(limit);
        self
    }

    /// Set the maximum tables in each Store.
    pub fn table_limit(mut self, limit: usize) -> Self {
        self.config.table_limit = Some(limit);
        self
    }

    /// Set the maximum linear memories in each Store.
    pub fn memory_count_limit(mut self, limit: usize) -> Self {
        self.config.memory_count_limit = Some(limit);
        self
    }

    /// Set the fuel budget reset before every invocation.
    pub fn fuel_per_call(mut self, fuel: u64) -> Self {
        self.config.fuel_per_call = Some(fuel);
        self
    }

    /// Set the relative epoch deadline reset before every invocation.
    pub fn epoch_deadline_ticks(mut self, ticks: u64) -> Self {
        self.config.epoch_deadline_ticks = Some(ticks);
        self
    }

    /// Set the WASI Preview1 capability policy.
    pub fn wasi_policy(mut self, policy: WasiPolicy) -> Self {
        self.config.wasi = policy;
        self
    }

    /// Compile the module and create a shareable runtime.
    pub fn build(self) -> Result<Runtime> {
        ensure!(
            self.config.fuel_per_call != Some(0),
            "fuel_per_call must be greater than zero"
        );
        ensure!(
            self.config.epoch_deadline_ticks != Some(0),
            "epoch_deadline_ticks must be greater than zero"
        );
        #[cfg(not(target_has_atomic = "64"))]
        ensure!(
            self.config.epoch_deadline_ticks.is_none(),
            "epoch interruption requires 64-bit atomics"
        );

        let mut engine_config = wasmtime::Config::new();
        engine_config.consume_fuel(self.config.fuel_per_call.is_some());
        engine_config.epoch_interruption(self.config.epoch_deadline_ticks.is_some());
        let engine = Engine::new(&engine_config).context("failed to create Wasmtime engine")?;
        Runtime::with_config_engine(self.binary, self.config, &engine)
    }
}

/// Implement `Clone` for `Runtime` so that we can simply `clone` to create a new
/// runtime for the same WASM binary.
impl Clone for Runtime {
    fn clone(&self) -> Self {
        Self {
            module: self.module.clone(), // this will share the immutable wasm binary
            config: self.config,
            functions: self.functions.clone(),
            types: self.types.clone(),
            instances: Default::default(), // just initialize a new instance pool
            abi_version: self.abi_version,
        }
    }
}

impl Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("config", &self.config)
            .field("functions", &self.functions)
            .field("types", &self.types)
            .field("instances", &self.instances.lock().unwrap().len())
            .finish()
    }
}

impl Runtime {
    /// Create a new UDF runtime from a WASM binary.
    ///
    /// This preserves the legacy unbounded resource defaults. Production
    /// embedders should use [`Runtime::builder`] and configure explicit limits.
    /// WASI clocks are denied and randomness is deterministic by default.
    pub fn new(binary: &[u8]) -> Result<Self> {
        Self::with_config(binary, Config::default())
    }

    /// Create a policy builder for a WASM binary.
    pub fn builder(binary: &[u8]) -> RuntimeBuilder<'_> {
        RuntimeBuilder::new(binary)
    }

    /// Create a new UDF runtime from a WASM binary with configuration.
    pub fn with_config(binary: &[u8], config: Config) -> Result<Self> {
        RuntimeBuilder::new(binary).config(config).build()
    }

    /// Create a new UDF runtime from a WASM binary with a customized engine.
    fn with_config_engine(binary: &[u8], config: Config, engine: &Engine) -> Result<Self> {
        let module = Module::from_binary(engine, binary).context("failed to load wasm binary")?;

        // check abi version
        let version = module
            .exports()
            .find_map(|e| e.name().strip_prefix("ARROWUDF_VERSION_"))
            .context("version not found")?;
        let (major, minor) = version.split_once('_').context("invalid version")?;
        let (major, minor) = (major.parse::<u8>()?, minor.parse::<u8>()?);
        ensure!(major <= 3, "unsupported abi version: {major}.{minor}");

        let mut functions = HashSet::new();
        let mut types = HashMap::new();
        for export in module.exports() {
            if let Some(encoded) = export.name().strip_prefix("arrowudf_") {
                let name = base64_decode(encoded).context("invalid symbol")?;
                functions.insert(name);
            } else if let Some(encoded) = export.name().strip_prefix("arrowudt_") {
                let meta = base64_decode(encoded).context("invalid symbol")?;
                let (name, fields) = meta.split_once('=').context("invalid type string")?;
                types.insert(name.to_string(), fields.to_string());
            }
        }

        Ok(Self {
            module,
            config,
            functions: functions.into(),
            types: types.into(),
            instances: Mutex::new(vec![]),
            abi_version: (major, minor),
        })
    }

    /// Return available functions.
    pub fn functions(&self) -> impl Iterator<Item = &str> {
        self.functions.iter().map(|s| s.as_str())
    }

    /// Return available types.
    pub fn types(&self) -> impl Iterator<Item = (&str, &str)> {
        self.types.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Return the ABI version.
    pub fn abi_version(&self) -> (u8, u8) {
        self.abi_version
    }

    /// Return a weak epoch controller when epoch interruption is enabled.
    ///
    /// The embedding application owns the ticker cadence and may share this
    /// controller across threads. Each Store keeps an independent relative
    /// deadline that is reset before every invocation.
    pub fn epoch_controller(&self) -> Option<EpochController> {
        self.config.epoch_deadline_ticks.map(|_| EpochController {
            engine: self.module.engine().weak(),
        })
    }

    /// Find a function by name, argument types and return type.
    /// The returned [`FunctionHandle`] can be used to call the function.
    pub fn find_function(
        &self,
        name: &str,
        arg_types: Vec<impl IntoField>,
        return_type: impl IntoField,
    ) -> Result<FunctionHandle> {
        // construct the function signature
        let args = arg_types
            .into_iter()
            .map(|x| x.into_field(""))
            .collect::<Vec<_>>();
        let ret = return_type.into_field("");
        let sig = function_signature_of(name, &args, &ret, false)?;
        // now find the export name
        if let Some(export_name) = self.get_function_export_name_by_inlined_signature(&sig) {
            return Ok(FunctionHandle {
                export_name: export_name.to_owned(),
                return_field: ret,
            });
        }
        bail!(
            "function not found in wasm binary: \"{}\"\nHINT: available functions:\n  {}\navailable types:\n  {}",
            sig,
            self.functions.iter().join("\n  "),
            self.types
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .join("\n  "),
        )
    }

    /// Find a table function by name, argument types and return type.
    /// The returned [`TableFunctionHandle`] can be used to call the table function.
    pub fn find_table_function(
        &self,
        name: &str,
        arg_types: Vec<impl IntoField>,
        return_type: impl IntoField,
    ) -> Result<TableFunctionHandle> {
        // construct the function signature
        let args = arg_types
            .into_iter()
            .map(|x| x.into_field(""))
            .collect::<Vec<_>>();
        let ret = return_type.into_field("");
        let sig = function_signature_of(name, &args, &ret, true)?;
        // now find the export name
        if let Some(export_name) = self.get_function_export_name_by_inlined_signature(&sig) {
            return Ok(TableFunctionHandle {
                export_name: export_name.to_owned(),
            });
        }
        bail!(
            "table function not found in wasm binary: \"{}\"\nHINT: available functions:\n  {}\navailable types:\n  {}",
            sig,
            self.functions.iter().join("\n  "),
            self.types
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .join("\n  "),
        )
    }

    /// Find the four functions that implement the opaque-state aggregate ABI.
    ///
    /// The ABI keeps aggregate state in host-owned Arrow binary values and
    /// uses list arguments to preserve one guest invocation per input batch:
    ///
    /// ```text
    /// <name>_init() -> binary
    /// <name>_update(binary, arg0[], ...) -> binary
    /// <name>_merge(binary, binary[]) -> binary
    /// <name>_finish(binary) -> result
    /// ```
    pub fn find_aggregate_function(
        &self,
        name: &str,
        arg_types: Vec<impl IntoField>,
        return_type: impl IntoField,
    ) -> Result<AggregateFunctionHandle> {
        let arguments = arg_types
            .into_iter()
            .enumerate()
            .map(|(index, argument)| argument.into_field(&format!("arg{index}")))
            .collect::<Vec<_>>();
        let return_field = return_type.into_field("result");
        let state_field = Field::new("state", DataType::Binary, false);
        let list_arguments = arguments
            .iter()
            .map(|argument| {
                DataType::List(Arc::new(Field::new(
                    "item",
                    argument.data_type().clone(),
                    true,
                )))
            })
            .collect::<Vec<_>>();

        Ok(AggregateFunctionHandle {
            init: self.find_function(
                &format!("{name}_init"),
                Vec::<Field>::new(),
                state_field.clone(),
            )?,
            update: self.find_function(
                &format!("{name}_update"),
                std::iter::once(DataType::Binary)
                    .chain(list_arguments)
                    .collect::<Vec<_>>(),
                state_field.clone(),
            )?,
            merge: self.find_function(
                &format!("{name}_merge"),
                vec![
                    DataType::Binary,
                    DataType::List(Arc::new(Field::new("item", DataType::Binary, true))),
                ],
                state_field.clone(),
            )?,
            finish: self.find_function(
                &format!("{name}_finish"),
                vec![DataType::Binary],
                return_field,
            )?,
            arguments,
        })
    }

    /// Create one host-owned aggregate state.
    pub fn create_aggregate_state(&self, function: &AggregateFunctionHandle) -> Result<Vec<u8>> {
        let input = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            Vec::new(),
            &RecordBatchOptions::default().with_row_count(Some(1)),
        )?;
        aggregate_state_from_output(self.call(&function.init, &input)?)
    }

    /// Update one aggregate state with a complete Arrow input batch.
    pub fn update_aggregate_state(
        &self,
        function: &AggregateFunctionHandle,
        state: &[u8],
        input: &RecordBatch,
    ) -> Result<Vec<u8>> {
        ensure!(
            input.num_columns() == function.arguments.len(),
            "aggregate input column-count mismatch: expected {}, got {}",
            function.arguments.len(),
            input.num_columns()
        );

        let mut fields = Vec::with_capacity(input.num_columns() + 1);
        let mut columns = Vec::with_capacity(input.num_columns() + 1);
        fields.push(Field::new("state", DataType::Binary, false));
        columns.push(Arc::new(BinaryArray::from(vec![Some(state)])) as ArrayRef);
        for (index, (argument, column)) in
            function.arguments.iter().zip(input.columns()).enumerate()
        {
            ensure!(
                argument.data_type() == column.data_type(),
                "aggregate input type mismatch at column {index}: expected {}, got {}",
                argument.data_type(),
                column.data_type()
            );
            let list = singleton_list(column.clone())?;
            fields.push(Field::new(
                format!("arg{index}"),
                list.data_type().clone(),
                false,
            ));
            columns.push(Arc::new(list));
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        aggregate_state_from_output(self.call(&function.update, &batch)?)
    }

    /// Merge a batch of partial states into one aggregate state.
    pub fn merge_aggregate_states(
        &self,
        function: &AggregateFunctionHandle,
        state: &[u8],
        partial_states: ArrayRef,
    ) -> Result<Vec<u8>> {
        ensure!(
            partial_states.data_type() == &DataType::Binary,
            "aggregate partial-state type mismatch: expected Binary, got {}",
            partial_states.data_type()
        );
        let states = singleton_list(partial_states)?;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("state", DataType::Binary, false),
                Field::new("partial_states", states.data_type().clone(), false),
            ])),
            vec![
                Arc::new(BinaryArray::from(vec![Some(state)])),
                Arc::new(states),
            ],
        )?;
        aggregate_state_from_output(self.call(&function.merge, &batch)?)
    }

    /// Finalize one or more aggregate states in a single guest invocation.
    pub fn finish_aggregate_states(
        &self,
        function: &AggregateFunctionHandle,
        states: ArrayRef,
    ) -> Result<RecordBatch> {
        ensure!(
            states.data_type() == &DataType::Binary,
            "aggregate state type mismatch: expected Binary, got {}",
            states.data_type()
        );
        let input = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "state",
                DataType::Binary,
                false,
            )])),
            vec![states],
        )?;
        self.call(&function.finish, &input)
    }

    /// Given a function signature that inlines struct types, find the export function name.
    ///
    /// # Example
    ///
    /// ```text
    /// types = { "KeyValue": "key:string,value:string" }
    /// input = "keyvalue(string, string) -> struct<key:string,value:string>"
    /// output = "keyvalue(string, string) -> struct KeyValue"
    /// ```
    fn get_function_export_name_by_inlined_signature(&self, s: &str) -> Option<&str> {
        if let Some(f) = self.functions.get(s) {
            return Some(f);
        }
        self.functions
            .iter()
            .find(|f| self.inline_types(f) == s)
            .map(|f| f.as_str())
    }

    /// Inline types in function signature.
    ///
    /// # Example
    ///
    /// ```text
    /// types = { "KeyValue": "key:string,value:string" }
    /// input = "keyvalue(string, string) -> struct KeyValue"
    /// output = "keyvalue(string, string) -> struct<key:string,value:string>"
    /// ```
    fn inline_types(&self, s: &str) -> String {
        let mut inlined = s.to_string();
        loop {
            let replaced = inlined.clone();
            for (k, v) in self.types.iter() {
                inlined = inlined.replace(&format!("struct {k}"), &format!("struct<{v}>"));
            }
            if replaced == inlined {
                return inlined;
            }
        }
    }

    /// Call a function.
    pub fn call(&self, func: &FunctionHandle, input: &RecordBatch) -> Result<RecordBatch> {
        let export_name = &func.export_name;
        if !self.functions.contains(export_name) {
            bail!("function not found: {export_name}");
        }

        // get an instance from the pool, or create a new one if the pool is empty
        let mut instance = if let Some(instance) = self.instances.lock().unwrap().pop() {
            instance
        } else {
            Instance::new(self)?
        };

        instance.reset_call_limits()?;

        // call the function
        let output = instance
            .call_scalar_function(export_name, input)
            .and_then(|output| {
                validate_scalar_output(&output, input.num_rows(), &func.return_field)?;
                Ok(output)
            });

        // put the instance back to the pool
        if output.is_ok() {
            self.instances.lock().unwrap().push(instance);
        }

        output
    }

    /// Call a table function.
    pub fn call_table_function<'a>(
        &'a self,
        func: &'a TableFunctionHandle,
        input: &'a RecordBatch,
    ) -> Result<impl Iterator<Item = Result<RecordBatch>> + 'a> {
        use genawaiter2::sync::r#gen as generator;
        use genawaiter2::yield_;

        let export_name = &func.export_name;
        if !self.functions.contains(export_name) {
            bail!("function not found: {export_name}");
        }

        // get an instance from the pool, or create a new one if the pool is empty
        let mut instance = if let Some(instance) = self.instances.lock().unwrap().pop() {
            instance
        } else {
            Instance::new(self)?
        };

        instance.reset_call_limits()?;

        Ok(generator!({
            // call the function
            let iter = match instance.call_table_function(export_name, input) {
                Ok(iter) => iter,
                Err(e) => {
                    yield_!(Err(e));
                    return;
                }
            };
            for output in iter {
                yield_!(output);
            }
            // put the instance back to the pool
            // FIXME: if the iterator is not consumed, the instance will be dropped
            self.instances.lock().unwrap().push(instance);
        })
        .into_iter())
    }
}

pub struct FunctionHandle {
    export_name: String,
    return_field: Field,
}

/// Validated handles for opaque host-state aggregate execution.
pub struct AggregateFunctionHandle {
    init: FunctionHandle,
    update: FunctionHandle,
    merge: FunctionHandle,
    finish: FunctionHandle,
    arguments: Vec<Field>,
}

pub struct TableFunctionHandle {
    export_name: String,
}

fn singleton_list(values: ArrayRef) -> Result<ListArray> {
    let length = i32::try_from(values.len()).context("aggregate input batch is too large")?;
    Ok(ListArray::new(
        Arc::new(Field::new("item", values.data_type().clone(), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, length])),
        values,
        None,
    ))
}

fn aggregate_state_from_output(output: RecordBatch) -> Result<Vec<u8>> {
    ensure!(
        output.num_columns() == 1 && output.num_rows() == 1,
        "aggregate state operation must return exactly one binary value"
    );
    let states = output
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .context("aggregate state operation returned a non-binary value")?;
    ensure!(
        !states.is_null(0),
        "aggregate state operation returned NULL"
    );
    Ok(states.value(0).to_vec())
}

struct Instance {
    // extern "C" fn(len: usize, align: usize) -> *mut u8
    alloc: TypedFunc<(u32, u32), u32>,
    // extern "C" fn(ptr: *mut u8, len: usize, align: usize)
    dealloc: TypedFunc<(u32, u32, u32), ()>,
    // extern "C" fn(iter: *mut RecordBatchIter, out: *mut CSlice)
    record_batch_iterator_next: TypedFunc<(u32, u32), ()>,
    // extern "C" fn(iter: *mut RecordBatchIter)
    record_batch_iterator_drop: TypedFunc<u32, ()>,
    // extern "C" fn(ptr: *const u8, len: usize, out: *mut CSlice) -> i32
    functions: HashMap<String, TypedFunc<(u32, u32, u32), i32>>,
    memory: Memory,
    store: Store<(WasiCtx, StoreLimits)>,
    stdout: RamFileRef,
    stderr: RamFileRef,
    fuel_per_call: Option<u64>,
    epoch_deadline_ticks: Option<u64>,
    wasi_random_policy: WasiRandomPolicy,
}

impl Instance {
    /// Create a new instance.
    fn new(rt: &Runtime) -> Result<Self> {
        let module = &rt.module;
        let engine = module.engine();
        let mut linker = Linker::new(engine);
        wasi_common::sync::add_to_linker(&mut linker, |(wasi, _)| wasi)?;

        // Create a capability-minimized WASI context and put it in a Store.
        // No directories, sockets, environment, arguments, or stdin are inherited.
        let file_size_limit = rt.config.file_size_limit.unwrap_or(1024);
        let stdout = RamFileRef::new(RamFile::with_size_limit(file_size_limit));
        let stderr = RamFileRef::new(RamFile::with_size_limit(file_size_limit));
        let random: Box<dyn wasi_common::RngCore + Send + Sync> = match rt.config.wasi.random {
            WasiRandomPolicy::Deterministic => deterministic_random(),
            WasiRandomPolicy::Ambient => random_ctx(),
        };
        let clocks = match rt.config.wasi.clocks {
            WasiClockPolicy::Denied => WasiClocks::new(),
            WasiClockPolicy::Ambient => clocks_ctx(),
        };
        let wasi = WasiCtx::new(random, clocks, Box::new(DeniedWasiScheduler), Table::new());
        wasi.set_stdout(Box::new(stdout.clone()));
        wasi.set_stderr(Box::new(stderr.clone()));
        let limits = {
            let mut builder = StoreLimitsBuilder::new();
            if let Some(limit) = rt.config.memory_size_limit {
                builder = builder.memory_size(limit);
            }
            if let Some(limit) = rt.config.table_elements_limit {
                builder = builder.table_elements(limit);
            }
            if let Some(limit) = rt.config.instance_limit {
                builder = builder.instances(limit);
            }
            if let Some(limit) = rt.config.table_limit {
                builder = builder.tables(limit);
            }
            if let Some(limit) = rt.config.memory_count_limit {
                builder = builder.memories(limit);
            }
            if rt.config.memory_size_limit.is_some() || rt.config.table_elements_limit.is_some() {
                builder = builder.trap_on_grow_failure(true);
            }
            builder.build()
        };
        let mut store = Store::new(engine, (wasi, limits));
        store.limiter(|(_, limiter)| limiter);

        if let Some(fuel) = rt.config.fuel_per_call {
            store
                .set_fuel(fuel)
                .context("failed to initialize Wasmtime fuel")?;
        }
        #[cfg(target_has_atomic = "64")]
        if let Some(ticks) = rt.config.epoch_deadline_ticks {
            store.set_epoch_deadline(ticks);
            store.epoch_deadline_trap();
        }

        let instance = linker
            .instantiate(&mut store, module)
            .map_err(classify_execution_error)
            .context("failed to instantiate WASM UDF module")?;
        let mut functions = HashMap::new();
        for export in module.exports() {
            let Some(encoded) = export.name().strip_prefix("arrowudf_") else {
                continue;
            };
            let name = base64_decode(encoded).context("invalid symbol")?;
            let func = instance.get_typed_func(&mut store, export.name())?;
            functions.insert(name, func);
        }
        let alloc = instance.get_typed_func(&mut store, "alloc")?;
        let dealloc = instance.get_typed_func(&mut store, "dealloc")?;
        let record_batch_iterator_next =
            instance.get_typed_func(&mut store, "record_batch_iterator_next")?;
        let record_batch_iterator_drop =
            instance.get_typed_func(&mut store, "record_batch_iterator_drop")?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .context("no memory")?;

        Ok(Instance {
            alloc,
            dealloc,
            record_batch_iterator_next,
            record_batch_iterator_drop,
            memory,
            store,
            functions,
            stdout,
            stderr,
            fuel_per_call: rt.config.fuel_per_call,
            epoch_deadline_ticks: rt.config.epoch_deadline_ticks,
            wasi_random_policy: rt.config.wasi.random,
        })
    }

    /// Reset invocation-scoped execution controls before entering guest code.
    fn reset_call_limits(&mut self) -> Result<()> {
        if self.wasi_random_policy == WasiRandomPolicy::Deterministic {
            let mut random = self
                .store
                .data_mut()
                .0
                .random
                .lock()
                .map_err(|_| anyhow!("deterministic WASI random state lock was poisoned"))?;
            *random = deterministic_random();
        }
        if let Some(fuel) = self.fuel_per_call {
            self.store
                .set_fuel(fuel)
                .context("failed to reset Wasmtime fuel for UDF invocation")?;
        }
        #[cfg(target_has_atomic = "64")]
        if let Some(ticks) = self.epoch_deadline_ticks {
            self.store.set_epoch_deadline(ticks);
            self.store.epoch_deadline_trap();
        }
        Ok(())
    }

    /// Call a scalar function.
    fn call_scalar_function(&mut self, name: &str, input: &RecordBatch) -> Result<RecordBatch> {
        // TODO: optimize data transfer
        // currently there are 3 copies in input path:
        //      host record batch -> host encoding -> wasm memory -> wasm record batch
        // and 2 copies in output path:
        //      wasm record batch -> wasm memory -> host record batch

        // get function
        let func = self
            .functions
            .get(name)
            .with_context(|| format!("function not found: {name}"))?;

        // encode input batch
        let input = encode_record_batch(input)?;

        // allocate memory for input buffer and output struct
        let alloc_len = u32::try_from(input.len() + 4 * 2).context("input too large")?;
        let alloc_ptr = self
            .alloc
            .call(&mut self.store, (alloc_len, 4))
            .map_err(classify_execution_error)?;
        ensure!(alloc_ptr != 0, "failed to allocate for input");
        let in_ptr = alloc_ptr
            .checked_add(4 * 2)
            .context("guest input pointer overflow")?;

        // write input to memory
        self.memory
            .write(&mut self.store, in_ptr as usize, &input)?;

        // call the function
        let result = func
            .call(&mut self.store, (in_ptr, input.len() as u32, alloc_ptr))
            .map_err(classify_execution_error);
        let errno = self.append_stdio(result)?;

        // get return values
        let out_ptr = self.read_u32(alloc_ptr)?;
        let out_len = self.read_u32(
            alloc_ptr
                .checked_add(4)
                .context("guest output-length pointer overflow")?,
        )?;

        // read output from memory
        let out_range = checked_u32_range(out_ptr, out_len)?;
        let out_bytes = self
            .memory
            .data(&self.store)
            .get(out_range)
            .context("output slice out of bounds")?;
        let result = match errno {
            0 => Ok(decode_record_batch(out_bytes)?),
            _ => Err(anyhow!("{}", std::str::from_utf8(out_bytes)?)),
        };

        // deallocate memory
        self.dealloc
            .call(&mut self.store, (alloc_ptr, alloc_len, 4))
            .map_err(classify_execution_error)?;
        self.dealloc
            .call(&mut self.store, (out_ptr, out_len, 1))
            .map_err(classify_execution_error)?;

        result
    }

    /// Call a table function.
    fn call_table_function<'a>(
        &'a mut self,
        name: &str,
        input: &RecordBatch,
    ) -> Result<impl Iterator<Item = Result<RecordBatch>> + 'a> {
        // TODO: optimize data transfer
        // currently there are 3 copies in input path:
        //      host record batch -> host encoding -> wasm memory -> wasm record batch
        // and 2 copies in output path:
        //      wasm record batch -> wasm memory -> host record batch

        // get function
        let func = self
            .functions
            .get(name)
            .with_context(|| format!("function not found: {name}"))?;

        // encode input batch
        let input = encode_record_batch(input)?;

        // allocate memory for input buffer and output struct
        let alloc_len = u32::try_from(input.len() + 4 * 2).context("input too large")?;
        let alloc_ptr = self
            .alloc
            .call(&mut self.store, (alloc_len, 4))
            .map_err(classify_execution_error)?;
        ensure!(alloc_ptr != 0, "failed to allocate for input");
        let in_ptr = alloc_ptr
            .checked_add(4 * 2)
            .context("guest input pointer overflow")?;

        // write input to memory
        self.memory
            .write(&mut self.store, in_ptr as usize, &input)?;

        // call the function
        let result = func
            .call(&mut self.store, (in_ptr, input.len() as u32, alloc_ptr))
            .map_err(classify_execution_error);
        let errno = self.append_stdio(result)?;

        // get return values
        let out_ptr = self.read_u32(alloc_ptr)?;
        let out_len = self.read_u32(
            alloc_ptr
                .checked_add(4)
                .context("guest output-length pointer overflow")?,
        )?;

        // read output from memory
        let out_range = checked_u32_range(out_ptr, out_len)?;
        let out_bytes = self
            .memory
            .data(&self.store)
            .get(out_range)
            .context("output slice out of bounds")?;

        let ptr = match errno {
            0 => out_ptr,
            _ => {
                let err = anyhow!("{}", std::str::from_utf8(out_bytes)?);
                // deallocate memory
                self.dealloc
                    .call(&mut self.store, (alloc_ptr, alloc_len, 4))
                    .map_err(classify_execution_error)?;
                self.dealloc
                    .call(&mut self.store, (out_ptr, out_len, 1))
                    .map_err(classify_execution_error)?;

                return Err(err);
            }
        };

        struct RecordBatchIter<'a> {
            instance: &'a mut Instance,
            ptr: u32,
            alloc_ptr: u32,
            alloc_len: u32,
        }

        impl RecordBatchIter<'_> {
            /// Get the next record batch.
            fn next(&mut self) -> Result<Option<RecordBatch>> {
                self.instance
                    .record_batch_iterator_next
                    .call(&mut self.instance.store, (self.ptr, self.alloc_ptr))
                    .map_err(classify_execution_error)?;
                // get return values
                let out_ptr = self.instance.read_u32(self.alloc_ptr)?;
                let out_len = self.instance.read_u32(
                    self.alloc_ptr
                        .checked_add(4)
                        .context("guest output-length pointer overflow")?,
                )?;

                if out_ptr == 0 {
                    // end of iteration
                    return Ok(None);
                }

                // read output from memory
                let out_range = checked_u32_range(out_ptr, out_len)?;
                let out_bytes = self
                    .instance
                    .memory
                    .data(&self.instance.store)
                    .get(out_range)
                    .context("output slice out of bounds")?;
                let batch = decode_record_batch(out_bytes)?;

                // dealloc output
                self.instance
                    .dealloc
                    .call(&mut self.instance.store, (out_ptr, out_len, 1))
                    .map_err(classify_execution_error)?;

                Ok(Some(batch))
            }
        }

        impl Iterator for RecordBatchIter<'_> {
            type Item = Result<RecordBatch>;

            fn next(&mut self) -> Option<Self::Item> {
                let result = self.next();
                self.instance.append_stdio(result).transpose()
            }
        }

        impl Drop for RecordBatchIter<'_> {
            fn drop(&mut self) {
                _ = self.instance.dealloc.call(
                    &mut self.instance.store,
                    (self.alloc_ptr, self.alloc_len, 4),
                );
                _ = self
                    .instance
                    .record_batch_iterator_drop
                    .call(&mut self.instance.store, self.ptr);
            }
        }

        Ok(RecordBatchIter {
            instance: self,
            ptr,
            alloc_ptr,
            alloc_len,
        })
    }

    /// Read a `u32` from memory.
    fn read_u32(&mut self, ptr: u32) -> Result<u32> {
        let bytes = self
            .memory
            .data(&self.store)
            .get(checked_u32_range(ptr, 4)?)
            .context("guest u32 pointer out of bounds")?;
        Ok(u32::from_le_bytes(
            bytes
                .try_into()
                .context("guest u32 value had an invalid length")?,
        ))
    }

    /// Take stdout and stderr, append to the error context.
    fn append_stdio<T>(&self, result: Result<T>) -> Result<T> {
        let stdout = self.stdout.take();
        let stderr = self.stderr.take();
        match result {
            Ok(v) => Ok(v),
            Err(e) => Err(e.context(format!(
                "--- stdout\n{}\n--- stderr\n{}",
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr),
            ))),
        }
    }
}

fn deterministic_random() -> Box<dyn wasi_common::RngCore + Send + Sync> {
    Box::new(Deterministic::new(b"arrow-udf deterministic WASI".to_vec()))
}

fn classify_execution_error(error: Error) -> Error {
    match error.downcast_ref::<Trap>() {
        Some(Trap::OutOfFuel) => error.context(RuntimeControlError::FuelExhausted),
        Some(Trap::Interrupt) => error.context(RuntimeControlError::EpochDeadlineExceeded),
        _ => error,
    }
}

fn checked_u32_range(ptr: u32, len: u32) -> Result<std::ops::Range<usize>> {
    let end = ptr
        .checked_add(len)
        .context("guest pointer plus length overflowed")?;
    Ok(ptr as usize..end as usize)
}

/// Decode a string from symbol name using customized base64.
fn base64_decode(input: &str) -> Result<String> {
    use base64::{
        Engine,
        alphabet::Alphabet,
        engine::{GeneralPurpose, general_purpose::NO_PAD},
    };
    // standard base64 uses '+' and '/', which is not a valid symbol name.
    // we use '$' and '_' instead.
    let alphabet =
        Alphabet::new("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$_").unwrap();
    let engine = GeneralPurpose::new(&alphabet, NO_PAD);
    let bytes = engine.decode(input)?;
    String::from_utf8(bytes).context("invalid utf8")
}

fn encode_record_batch(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = vec![];
    let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut buf, &batch.schema())?;
    writer.write(batch)?;
    writer.finish()?;
    drop(writer);
    Ok(buf)
}

fn decode_record_batch(bytes: &[u8]) -> Result<RecordBatch> {
    let mut reader = arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(bytes), None)?;
    let batch = reader
        .next()
        .context("guest returned Arrow IPC without a record batch")??;
    Ok(batch)
}

fn validate_scalar_output(
    output: &RecordBatch,
    expected_rows: usize,
    expected: &Field,
) -> Result<()> {
    let field = output
        .schema_ref()
        .fields()
        .first()
        .context("guest scalar output has no value column")?;
    ensure!(
        field.data_type() == expected.data_type(),
        "guest scalar output type mismatch: expected {}, got {}",
        expected.data_type(),
        field.data_type()
    );
    ensure!(
        output.num_rows() == expected_rows,
        "guest scalar output row-count mismatch: expected {expected_rows}, got {}",
        output.num_rows()
    );
    Ok(())
}

/// Construct the function signature generated by `function` macro.
fn function_signature_of(
    name: &str,
    args: &[Field],
    ret: &Field,
    is_table_function: bool,
) -> Result<String> {
    let args: Vec<_> = args.iter().map(field_to_typename).try_collect()?;
    let ret = field_to_typename(ret)?;
    Ok(format!(
        "{}({}){}{}",
        name,
        args.iter().join(","),
        if is_table_function { "->>" } else { "->" },
        ret
    ))
}

fn field_to_typename(field: &Field) -> Result<String> {
    if let Some(ext_typename) = field.metadata().get("ARROW:extension:name")
        && let Some(typename) = ext_typename.strip_prefix("arrowudf.")
    {
        // cases like "arrowudf.decimal" and "arrowudf.json"
        return Ok(typename.to_owned());
    }
    let ty = field.data_type();
    // Convert arrow types to type names in `arrow-udf-macros` according
    // to `arrow_udf_macros::types::TYPE_MATRIX`.
    Ok(match ty {
        DataType::Null => "null".to_owned(),
        DataType::Boolean => "boolean".to_owned(),
        DataType::Int8 => "int8".to_owned(),
        DataType::Int16 => "int16".to_owned(),
        DataType::Int32 => "int32".to_owned(),
        DataType::Int64 => "int64".to_owned(),
        DataType::UInt8 => "uint8".to_owned(),
        DataType::UInt16 => "uint16".to_owned(),
        DataType::UInt32 => "uint32".to_owned(),
        DataType::UInt64 => "uint64".to_owned(),
        DataType::Float32 => "float32".to_owned(),
        DataType::Float64 => "float64".to_owned(),
        DataType::Date32 => "date32".to_owned(),
        DataType::Time64(TimeUnit::Microsecond) => "time64".to_owned(),
        DataType::Timestamp(TimeUnit::Microsecond, None) => "timestamp".to_owned(),
        DataType::Interval(IntervalUnit::MonthDayNano) => "interval".to_owned(),
        DataType::Utf8 => "string".to_owned(),
        DataType::Binary => "binary".to_owned(),
        DataType::LargeUtf8 => "largestring".to_owned(),
        DataType::LargeBinary => "largebinary".to_owned(),
        DataType::List(elem) => format!("{}[]", field_to_typename(elem)?),
        DataType::Struct(fields) => {
            let fields: Vec<_> = fields
                .iter()
                .map(|x| Ok::<_, anyhow::Error>((x.name(), field_to_typename(x)?)))
                .try_collect()?;
            format!(
                "struct<{}>",
                fields
                    .iter()
                    .map(|(name, typename)| format!("{name}:{typename}"))
                    .join(",")
            )
        }
        _ => {
            bail!("unsupported data type: {ty}");
        }
    })
}
