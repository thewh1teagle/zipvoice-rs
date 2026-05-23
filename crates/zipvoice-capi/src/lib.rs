use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use zipvoice_rs::{CreateOptions, ZipVoice, write_wav_mono_16bit};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub struct ZipVoiceModel {
    inner: ZipVoice,
}

fn set_error(message: impl ToString) -> c_int {
    let message = message.to_string().replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(message).ok();
    });
    -1
}

fn clear_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

unsafe fn cstr_arg<'a>(ptr: *const c_char, name: &str) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err(format!("{name} is null"));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|err| format!("{name} is not valid UTF-8: {err}"))
}

#[unsafe(no_mangle)]
pub extern "C" fn zipvoice_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|err| err.as_ptr())
            .unwrap_or(ptr::null())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipvoice_model_load(
    zipvoice_path: *const c_char,
    vocos_path: *const c_char,
) -> *mut ZipVoiceModel {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let zipvoice_path = unsafe { cstr_arg(zipvoice_path, "zipvoice_path") }?;
        let vocos_path = unsafe { cstr_arg(vocos_path, "vocos_path") }?;
        ZipVoice::load_with_vocos(zipvoice_path, vocos_path)
            .map(|inner| Box::into_raw(Box::new(ZipVoiceModel { inner })))
            .map_err(|err| err.to_string())
    }));
    match result {
        Ok(Ok(model)) => model,
        Ok(Err(err)) => {
            set_error(err);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic while loading model");
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipvoice_model_free(model: *mut ZipVoiceModel) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zipvoice_generate_wav(
    model: *mut ZipVoiceModel,
    ref_wav: *const c_char,
    ref_phonemes: *const c_char,
    target_phonemes: *const c_char,
    speed: f32,
    num_steps: usize,
    t_shift: f32,
    guidance_scale: f32,
    seed: u64,
    verbose: bool,
    output_path: *const c_char,
) -> c_int {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if model.is_null() {
            return Err("model is null".to_string());
        }
        let ref_wav = unsafe { cstr_arg(ref_wav, "ref_wav") }?;
        let ref_phonemes = unsafe { cstr_arg(ref_phonemes, "ref_phonemes") }?;
        let target_phonemes = unsafe { cstr_arg(target_phonemes, "target_phonemes") }?;
        let output_path = unsafe { cstr_arg(output_path, "output_path") }?;
        let options = CreateOptions {
            speed,
            num_steps,
            t_shift,
            guidance_scale,
            seed,
            verbose,
        };
        let (samples, sample_rate) = unsafe { &mut *model }
            .inner
            .create_with_options(ref_wav, ref_phonemes, target_phonemes, options)
            .map_err(|err| err.to_string())?;
        write_wav_mono_16bit(output_path, &samples, sample_rate).map_err(|err| err.to_string())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => set_error(err),
        Err(_) => set_error("panic while generating audio"),
    }
}
