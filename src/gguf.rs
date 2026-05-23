use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use half::f16;
use llama_rs_sys as ffi;
use thiserror::Error;

const SCHED_GRAPH_NODES: usize = 65_536;

static INIT_LOGGING: Once = Once::new();
static GGML_VERBOSE: AtomicBool = AtomicBool::new(false);

pub fn set_ggml_verbose(verbose: bool) {
    init_ggml_logging();
    GGML_VERBOSE.store(verbose, Ordering::Relaxed);
}

fn init_ggml_logging() {
    INIT_LOGGING.call_once(|| unsafe {
        ffi::ggml_log_set(Some(ggml_log_callback), ptr::null_mut());
    });
}

unsafe extern "C" fn ggml_log_callback(
    _level: ffi::ggml_log_level,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    if GGML_VERBOSE.load(Ordering::Relaxed) && !text.is_null() {
        eprint!("{}", unsafe { CStr::from_ptr(text) }.to_string_lossy());
    }
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("path contains nul byte: {0}")]
    NulPath(PathBuf),
    #[error("invalid key: {0}")]
    InvalidKey(String),
    #[error("failed to open GGUF model: {0}")]
    Open(PathBuf),
    #[error("missing GGUF tensor: {0}")]
    MissingTensor(String),
    #[error("missing GGUF tensor name at index {0}")]
    MissingTensorName(i64),
    #[error("invalid GGUF tensor index: {0}")]
    TensorIndex(i64),
    #[error("invalid tensor range in {path}: offset={offset} size={size}")]
    InvalidTensorRange {
        path: PathBuf,
        offset: usize,
        size: usize,
    },
    #[error("unsupported tensor type for {name}: {tensor_type}")]
    UnsupportedTensorType {
        name: String,
        tensor_type: ffi::ggml_type,
    },
    #[error("ggml error: {0}")]
    Ggml(String),
}

pub type Result<T> = std::result::Result<T, GgufError>;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub index: i64,
    pub name: String,
    pub tensor_type: ffi::ggml_type,
    pub offset: usize,
    pub size: usize,
}

pub struct GgufModel {
    ctx: *mut ffi::gguf_context,
    meta_ctx: *mut ffi::ggml_context,
    path: PathBuf,
}

impl GgufModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        init_ggml_logging();
        let path = path.as_ref();
        let c_path = path_to_cstring(path)?;
        let mut meta_ctx = ptr::null_mut();
        let params = ffi::gguf_init_params {
            no_alloc: true,
            ctx: &mut meta_ctx,
        };
        let ctx = unsafe { ffi::gguf_init_from_file(c_path.as_ptr(), params) };
        if ctx.is_null() {
            return Err(GgufError::Open(path.to_path_buf()));
        }
        Ok(Self {
            ctx,
            meta_ctx,
            path: path.to_path_buf(),
        })
    }

    pub fn tensor_count(&self) -> i64 {
        unsafe { ffi::gguf_get_n_tensors(self.ctx) }
    }

    pub fn tensor(&self, index: i64) -> Result<TensorInfo> {
        if index < 0 || index >= self.tensor_count() {
            return Err(GgufError::TensorIndex(index));
        }
        let name = unsafe { ffi::gguf_get_tensor_name(self.ctx, index) };
        if name.is_null() {
            return Err(GgufError::MissingTensorName(index));
        }
        let name = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .to_string();
        Ok(TensorInfo {
            index,
            name,
            tensor_type: unsafe { ffi::gguf_get_tensor_type(self.ctx, index) },
            offset: unsafe { ffi::gguf_get_tensor_offset(self.ctx, index) },
            size: unsafe { ffi::gguf_get_tensor_size(self.ctx, index) },
        })
    }

    pub fn tensors(&self) -> impl Iterator<Item = Result<TensorInfo>> + '_ {
        (0..self.tensor_count()).map(|index| self.tensor(index))
    }

    pub fn tensor_by_name(&self, name: &str) -> Result<Option<TensorInfo>> {
        let c_name = CString::new(name).map_err(|_| GgufError::InvalidKey(name.into()))?;
        let index = unsafe { ffi::gguf_find_tensor(self.ctx, c_name.as_ptr()) };
        if index < 0 {
            return Ok(None);
        }
        self.tensor(index).map(Some)
    }

    pub fn tensor_shape_by_name(&self, name: &str) -> Result<Option<Vec<usize>>> {
        let c_name = CString::new(name).map_err(|_| GgufError::InvalidKey(name.into()))?;
        let tensor = unsafe { ffi::ggml_get_tensor(self.meta_ctx, c_name.as_ptr()) };
        if tensor.is_null() {
            return Ok(None);
        }
        let tensor = unsafe { &*tensor };
        let n_dims = unsafe { ffi::ggml_n_dims(tensor) as usize };
        let mut shape = Vec::new();
        for idx in 0..n_dims {
            shape.push(tensor.ne[idx] as usize);
        }
        Ok(Some(shape))
    }

    pub fn tensor_bytes(&self, index: i64) -> Result<Vec<u8>> {
        let tensor = self.tensor(index)?;
        let absolute_offset = self
            .data_offset()
            .checked_add(tensor.offset)
            .ok_or_else(|| GgufError::InvalidTensorRange {
                path: self.path.clone(),
                offset: tensor.offset,
                size: tensor.size,
            })?;
        let mut file = File::open(&self.path)?;
        let end = absolute_offset.checked_add(tensor.size).ok_or_else(|| {
            GgufError::InvalidTensorRange {
                path: self.path.clone(),
                offset: absolute_offset,
                size: tensor.size,
            }
        })?;
        if end > file.metadata()?.len() as usize {
            return Err(GgufError::InvalidTensorRange {
                path: self.path.clone(),
                offset: absolute_offset,
                size: tensor.size,
            });
        }
        let mut bytes = vec![0_u8; tensor.size];
        file.seek(SeekFrom::Start(absolute_offset as u64))?;
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub fn tensor_f32_by_name(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensor_by_name(name)?
            .ok_or_else(|| GgufError::MissingTensor(name.into()))?;
        let bytes = self.tensor_bytes(tensor.index)?;
        match tensor.tensor_type {
            ffi::ggml_type_GGML_TYPE_F32 => {
                bytes_to_f32(&bytes).ok_or_else(|| GgufError::InvalidTensorRange {
                    path: self.path.clone(),
                    offset: tensor.offset,
                    size: tensor.size,
                })
            }
            ffi::ggml_type_GGML_TYPE_F16 => {
                if bytes.len() % 2 != 0 {
                    return Err(GgufError::InvalidTensorRange {
                        path: self.path.clone(),
                        offset: tensor.offset,
                        size: tensor.size,
                    });
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                    .collect())
            }
            tensor_type => self.dequantize_tensor_to_f32(name, tensor_type, &bytes),
        }
    }

    fn dequantize_tensor_to_f32(
        &self,
        name: &str,
        tensor_type: ffi::ggml_type,
        bytes: &[u8],
    ) -> Result<Vec<f32>> {
        let shape = self
            .tensor_shape_by_name(name)?
            .ok_or_else(|| GgufError::MissingTensor(name.into()))?;
        let elements = shape.iter().product::<usize>();
        let traits = unsafe { ffi::ggml_get_type_traits(tensor_type) };
        if traits.is_null() {
            return Err(GgufError::UnsupportedTensorType {
                name: name.into(),
                tensor_type,
            });
        }
        let to_float =
            unsafe { (*traits).to_float }.ok_or_else(|| GgufError::UnsupportedTensorType {
                name: name.into(),
                tensor_type,
            })?;
        let expected = unsafe { ffi::ggml_row_size(tensor_type, elements as i64) };
        if bytes.len() != expected {
            return Err(GgufError::InvalidTensorRange {
                path: self.path.clone(),
                offset: 0,
                size: bytes.len(),
            });
        }
        let mut output = vec![0.0_f32; elements];
        unsafe {
            to_float(bytes.as_ptr().cast(), output.as_mut_ptr(), elements as i64);
        }
        Ok(output)
    }

    pub fn get_string(&self, key: &str) -> Result<Option<String>> {
        let c_key = CString::new(key).map_err(|_| GgufError::InvalidKey(key.into()))?;
        let index = unsafe { ffi::gguf_find_key(self.ctx, c_key.as_ptr()) };
        if index < 0 {
            return Ok(None);
        }
        let value = unsafe { ffi::gguf_get_val_str(self.ctx, index) };
        if value.is_null() {
            return Ok(None);
        }
        Ok(Some(
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .to_string(),
        ))
    }

    pub fn get_u32(&self, key: &str) -> Result<Option<u32>> {
        let c_key = CString::new(key).map_err(|_| GgufError::InvalidKey(key.into()))?;
        let index = unsafe { ffi::gguf_find_key(self.ctx, c_key.as_ptr()) };
        if index < 0 {
            return Ok(None);
        }
        Ok(Some(unsafe { ffi::gguf_get_val_u32(self.ctx, index) }))
    }

    fn data_offset(&self) -> usize {
        unsafe { ffi::gguf_get_data_offset(self.ctx) }
    }
}

impl Drop for GgufModel {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                ffi::gguf_free(self.ctx);
            }
            if !self.meta_ctx.is_null() {
                ffi::ggml_free(self.meta_ctx);
            }
        }
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    )
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes()).map_err(|_| GgufError::NulPath(path.into()))
}

pub struct GgmlWeights {
    ctx: *mut ffi::ggml_context,
    backend: ffi::ggml_backend_t,
    backend_cpu: ffi::ggml_backend_t,
    sched: ffi::ggml_backend_sched_t,
    buffer: ffi::ggml_backend_buffer_t,
    compute_meta: Vec<u8>,
    tensors: HashMap<String, *mut ffi::ggml_tensor>,
}

impl GgmlWeights {
    pub fn load_all(model: &GgufModel) -> Result<Self> {
        let specs = model
            .tensors()
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|info| {
                let shape = model
                    .tensor_shape_by_name(&info.name)?
                    .ok_or_else(|| GgufError::MissingTensor(info.name.clone()))?;
                Ok(TensorSpec {
                    name: info.name,
                    tensor_type: info.tensor_type,
                    shape,
                    index: info.index,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut weights = Self::new(specs.len())?;
        for spec in &specs {
            weights.create_tensor(spec)?;
        }
        weights.allocate_buffer()?;
        weights.load_tensor_data(model, &specs)?;
        Ok(weights)
    }

    pub fn tensor(&self, name: &str) -> Result<*mut ffi::ggml_tensor> {
        self.tensors
            .get(name)
            .copied()
            .ok_or_else(|| GgufError::MissingTensor(name.into()))
    }

    pub fn backend(&self) -> ffi::ggml_backend_t {
        self.backend
    }

    pub fn scheduler(&self) -> ffi::ggml_backend_sched_t {
        self.sched
    }

    pub fn backend_name(&self) -> String {
        unsafe {
            let name = ffi::ggml_backend_name(self.backend);
            if name.is_null() {
                return "unknown".into();
            }
            CStr::from_ptr(name).to_string_lossy().into_owned()
        }
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn compute_context(&mut self) -> Result<*mut ffi::ggml_context> {
        let params = ffi::ggml_init_params {
            mem_size: self.compute_meta.len(),
            mem_buffer: self.compute_meta.as_mut_ptr().cast(),
            no_alloc: true,
        };
        let ctx = unsafe { ffi::ggml_init(params) };
        if ctx.is_null() {
            return Err(GgufError::Ggml(
                "failed to initialize GGML compute context".into(),
            ));
        }
        Ok(ctx)
    }

    pub fn alloc_graph(&self, graph: *mut ffi::ggml_cgraph) -> Result<()> {
        let ok = unsafe { ffi::ggml_backend_sched_alloc_graph(self.sched, graph) };
        if !ok {
            return Err(GgufError::Ggml(
                "failed to allocate GGML scheduler graph".into(),
            ));
        }
        Ok(())
    }

    pub fn compute_graph(&self, graph: *mut ffi::ggml_cgraph) -> Result<()> {
        let status = unsafe { ffi::ggml_backend_sched_graph_compute(self.sched, graph) };
        if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
            return Err(GgufError::Ggml(format!(
                "GGML scheduler graph failed with status={status}"
            )));
        }
        Ok(())
    }

    pub fn reset_scheduler(&self) {
        unsafe {
            ffi::ggml_backend_sched_reset(self.sched);
        }
    }

    fn new(n_tensors: usize) -> Result<Self> {
        unsafe {
            if std::env::var_os("GGML_VK_PREFER_HOST_MEMORY").is_none()
                && std::env::var_os("GGML_VK_DISABLE_PREFER_HOST_MEMORY").is_none()
            {
                std::env::set_var("GGML_VK_PREFER_HOST_MEMORY", "1");
            }
            if std::env::var_os("GGML_VK_DISABLE_GRAPH_OPTIMIZE").is_none()
                && std::env::var_os("GGML_VK_ENABLE_GRAPH_OPTIMIZE").is_none()
            {
                std::env::set_var("GGML_VK_DISABLE_GRAPH_OPTIMIZE", "1");
            }
            ffi::ggml_backend_load_all();
            let backend = if std::env::var_os("ZIPVOICE_FORCE_CPU").is_some() {
                ffi::ggml_backend_init_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                    ptr::null(),
                )
            } else {
                ffi::ggml_backend_init_best()
            };
            if backend.is_null() {
                return Err(GgufError::Ggml("failed to initialize GGML backend".into()));
            }
            let params = ffi::ggml_init_params {
                mem_size: n_tensors * ffi::ggml_tensor_overhead(),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                ffi::ggml_backend_free(backend);
                return Err(GgufError::Ggml("failed to initialize GGML context".into()));
            }
            Ok(Self {
                ctx,
                backend,
                backend_cpu: ptr::null_mut(),
                sched: ptr::null_mut(),
                buffer: ptr::null_mut(),
                compute_meta: Vec::new(),
                tensors: HashMap::new(),
            })
        }
    }

    fn create_tensor(&mut self, spec: &TensorSpec) -> Result<()> {
        let c_name = CString::new(spec.name.as_str())
            .map_err(|_| GgufError::InvalidKey(spec.name.clone()))?;
        let mut ne = [1_i64; 4];
        for (idx, dim) in spec.shape.iter().enumerate().take(4) {
            ne[idx] = *dim as i64;
        }
        let n_dims = spec.shape.len().max(1).min(4) as i32;
        let tensor =
            unsafe { ffi::ggml_new_tensor(self.ctx, spec.tensor_type, n_dims, ne.as_ptr()) };
        if tensor.is_null() {
            return Err(GgufError::Ggml(format!(
                "failed to create tensor {}",
                spec.name
            )));
        }
        unsafe {
            ffi::ggml_set_name(tensor, c_name.as_ptr());
        }
        self.tensors.insert(spec.name.clone(), tensor);
        Ok(())
    }

    fn allocate_buffer(&mut self) -> Result<()> {
        self.buffer = unsafe { ffi::ggml_backend_alloc_ctx_tensors(self.ctx, self.backend) };
        if self.buffer.is_null() {
            return Err(GgufError::Ggml(
                "failed to allocate GGML weight buffer".into(),
            ));
        }
        unsafe {
            ffi::ggml_backend_buffer_set_usage(
                self.buffer,
                ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            );
        }
        Ok(())
    }

    fn load_tensor_data(&mut self, model: &GgufModel, specs: &[TensorSpec]) -> Result<()> {
        for spec in specs {
            let tensor = self.tensor(&spec.name)?;
            let bytes = model.tensor_bytes(spec.index)?;
            let expected = unsafe { ffi::ggml_nbytes(tensor) };
            if bytes.len() != expected {
                return Err(GgufError::Ggml(format!(
                    "tensor {} data has {} bytes, GGML expects {}",
                    spec.name,
                    bytes.len(),
                    expected
                )));
            }
            unsafe {
                ffi::ggml_backend_tensor_set(tensor, bytes.as_ptr().cast(), 0, bytes.len());
            }
        }
        unsafe {
            ffi::ggml_backend_synchronize(self.backend);
        }
        self.init_scheduler()?;
        Ok(())
    }

    fn init_scheduler(&mut self) -> Result<()> {
        unsafe {
            let device = ffi::ggml_backend_get_device(self.backend);
            let mut backends = vec![self.backend];
            if !device.is_null()
                && ffi::ggml_backend_dev_type(device)
                    != ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU
            {
                self.backend_cpu = ffi::ggml_backend_init_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                    ptr::null(),
                );
                if !self.backend_cpu.is_null() {
                    backends.push(self.backend_cpu);
                }
            }
            self.sched = ffi::ggml_backend_sched_new(
                backends.as_mut_ptr(),
                ptr::null_mut(),
                backends.len() as i32,
                SCHED_GRAPH_NODES,
                false,
                true,
            );
            if self.sched.is_null() {
                return Err(GgufError::Ggml(
                    "failed to initialize GGML backend scheduler".into(),
                ));
            }
            self.compute_meta = vec![
                0;
                ffi::ggml_tensor_overhead() * SCHED_GRAPH_NODES
                    + ffi::ggml_graph_overhead_custom(SCHED_GRAPH_NODES, false)
            ];
            Ok(())
        }
    }
}

impl Drop for GgmlWeights {
    fn drop(&mut self) {
        unsafe {
            if !self.sched.is_null() {
                ffi::ggml_backend_sched_free(self.sched);
            }
            if !self.buffer.is_null() {
                ffi::ggml_backend_buffer_free(self.buffer);
            }
            if !self.ctx.is_null() {
                ffi::ggml_free(self.ctx);
            }
            if !self.backend_cpu.is_null() {
                ffi::ggml_backend_free(self.backend_cpu);
            }
            if !self.backend.is_null() {
                ffi::ggml_backend_free(self.backend);
            }
        }
    }
}

struct TensorSpec {
    name: String,
    tensor_type: ffi::ggml_type,
    shape: Vec<usize>,
    index: i64,
}
