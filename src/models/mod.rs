pub mod llama;
pub mod onnx;
#[cfg(not(any(windows, target_os = "macos")))]
pub mod burn_decoder;
#[cfg(not(any(windows, target_os = "macos")))]
mod burn_decoder_generated;
