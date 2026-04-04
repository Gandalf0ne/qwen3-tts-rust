use crate::models::burn_decoder_generated;
use crate::models::onnx::DecoderState;
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_onnx::ModelGen;
use ndarray::{Array3, Array4};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

type BurnBackend = burn_wgpu::Wgpu<f32, i64, u32>;
type BurnDevice = burn_wgpu::WgpuDevice;
type BurnModel = burn_decoder_generated::Model<BurnBackend>;

const PREPARE_SCRIPT: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/prepare_decoder_onnx.py"));
const PREPARE_SCRIPT_NAME: &str = "prepare_decoder_onnx.py";
const CACHE_DIR_NAME: &str = "burn_vulkan";
const NORMALIZED_MODEL_NAME: &str = "qwen3_tts_decoder.inferred.nosplit.onnx";
const BURNPACK_NAME: &str = "qwen3_tts_decoder.inferred.bpk";
const GENERATED_RS_NAME: &str = "qwen3_tts_decoder.inferred.rs";

pub struct BurnAudioDecoder {
    device: BurnDevice,
    model: BurnModel,
}

impl BurnAudioDecoder {
    pub fn load(model_path: &str) -> Result<Self, Box<dyn Error>> {
        let burnpack_path = ensure_decoder_burnpack(Path::new(model_path))?;
        let device = BurnDevice::default();

        println!(
            "  [Burn] AudioDecoder: Initializing WGPU on Vulkan backend (device: {:?})",
            device
        );
        burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(&device, Default::default());

        let burnpack_path = burnpack_path
            .to_str()
            .ok_or("Decoder burnpack path contains invalid UTF-8")?;
        let model = BurnModel::from_file(burnpack_path, &device);

        println!("  [Burn] AudioDecoder: Loaded {}", burnpack_path);

        Ok(Self { device, model })
    }

    pub fn decode(
        &mut self,
        codes: &[i64],
        state: &mut DecoderState,
        is_final: bool,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let n_frames = codes.len() / 16;
        if n_frames == 0 && !is_final {
            return Ok(vec![]);
        }

        let audio_codes = int_tensor(codes.to_vec(), [1, n_frames, 16], &self.device);
        let is_last = float_tensor(vec![if is_final { 1.0 } else { 0.0 }], [1], &self.device);
        let pre_conv_history = array3_tensor(&state.pre_conv_history, &self.device);
        let latent_buffer = array3_tensor(&state.latent_buffer, &self.device);
        let conv_history = array3_tensor(&state.conv_history, &self.device);

        let past_keys: Vec<_> = state
            .kv_cache
            .iter()
            .map(|(k, _)| array4_tensor(k, &self.device))
            .collect();
        let past_values: Vec<_> = state
            .kv_cache
            .iter()
            .map(|(_, v)| array4_tensor(v, &self.device))
            .collect();

        let (
            final_wav,
            valid_samples,
            next_pre_conv_history,
            next_latent_buffer,
            next_conv_history,
            next_key_0,
            next_key_1,
            next_key_2,
            next_key_3,
            next_key_4,
            next_key_5,
            next_key_6,
            next_key_7,
            next_value_0,
            next_value_1,
            next_value_2,
            next_value_3,
            next_value_4,
            next_value_5,
            next_value_6,
            next_value_7,
        ) = self.model.forward(
            audio_codes,
            is_last,
            pre_conv_history,
            latent_buffer,
            conv_history,
            past_keys[0].clone(),
            past_keys[1].clone(),
            past_keys[2].clone(),
            past_keys[3].clone(),
            past_keys[4].clone(),
            past_keys[5].clone(),
            past_keys[6].clone(),
            past_keys[7].clone(),
            past_values[0].clone(),
            past_values[1].clone(),
            past_values[2].clone(),
            past_values[3].clone(),
            past_values[4].clone(),
            past_values[5].clone(),
            past_values[6].clone(),
            past_values[7].clone(),
        );

        let valid_samples = valid_samples.to_data().to_vec::<i64>()?;
        let valid_count = valid_samples
            .first()
            .copied()
            .unwrap_or_default()
            .max(0) as usize;

        let mut audio = final_wav.to_data().to_vec::<f32>()?;
        audio.truncate(valid_count);

        state.pre_conv_history = tensor_to_array3(next_pre_conv_history)?;
        state.latent_buffer = tensor_to_array3(next_latent_buffer)?;
        state.conv_history = tensor_to_array3(next_conv_history)?;

        let next_keys = vec![
            next_key_0, next_key_1, next_key_2, next_key_3, next_key_4, next_key_5, next_key_6,
            next_key_7,
        ];
        let next_values = vec![
            next_value_0,
            next_value_1,
            next_value_2,
            next_value_3,
            next_value_4,
            next_value_5,
            next_value_6,
            next_value_7,
        ];

        for (index, (next_key, next_value)) in next_keys
            .into_iter()
            .zip(next_values.into_iter())
            .enumerate()
        {
            state.kv_cache[index] = (tensor_to_array4(next_key)?, tensor_to_array4(next_value)?);
        }

        Ok(audio)
    }
}

fn ensure_decoder_burnpack(model_path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let onnx_dir = model_path
        .parent()
        .ok_or("Decoder ONNX path does not have a parent directory")?;
    let cache_dir = onnx_dir.join(CACHE_DIR_NAME);
    fs::create_dir_all(&cache_dir)?;

    let normalized_model_path = cache_dir.join(NORMALIZED_MODEL_NAME);
    let burnpack_path = cache_dir.join(BURNPACK_NAME);

    if burnpack_path.exists() {
        return Ok(burnpack_path);
    }

    let script_path = cache_dir.join(PREPARE_SCRIPT_NAME);
    ensure_prepare_script(&script_path)?;

    if !normalized_model_path.exists() {
        println!(
            "  [Burn] AudioDecoder: Preparing normalized ONNX graph at {}",
            normalized_model_path.display()
        );
        run_prepare_script(&script_path, model_path, &normalized_model_path)?;
    }

    println!(
        "  [Burn] AudioDecoder: Generating burnpack at {}",
        burnpack_path.display()
    );
    run_burn_codegen(&cache_dir, &normalized_model_path)?;

    if !burnpack_path.exists() {
        return Err(format!(
            "Burn decoder generation finished without creating {}",
            burnpack_path.display()
        )
        .into());
    }

    let generated_rs_path = cache_dir.join(GENERATED_RS_NAME);
    if generated_rs_path.exists() {
        let _ = fs::remove_file(generated_rs_path);
    }

    Ok(burnpack_path)
}

fn ensure_prepare_script(script_path: &Path) -> Result<(), Box<dyn Error>> {
    let write_script = match fs::read_to_string(script_path) {
        Ok(existing) => existing != PREPARE_SCRIPT,
        Err(_) => true,
    };

    if write_script {
        fs::write(script_path, PREPARE_SCRIPT)?;
    }

    Ok(())
}

fn run_prepare_script(
    script_path: &Path,
    input_path: &Path,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let output = Command::new("python3")
        .arg(script_path)
        .arg(input_path)
        .arg(output_path)
        .output()?;

    if output.status.success() {
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "Python decoder prep failed. Ensure `python3` and the `onnx` package are installed.\nstdout:\n{}\nstderr:\n{}",
        stdout, stderr
    )
    .into())
}

fn run_burn_codegen(cache_dir: &Path, normalized_model_path: &Path) -> Result<(), Box<dyn Error>> {
    let cache_dir = cache_dir
        .to_str()
        .ok_or("Burn cache directory contains invalid UTF-8")?;
    let normalized_model_path = normalized_model_path
        .to_str()
        .ok_or("Normalized decoder ONNX path contains invalid UTF-8")?;

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut model_gen = ModelGen::default();
        model_gen.input(normalized_model_path).out_dir(cache_dir);
        model_gen.run_from_cli();
    }));

    if result.is_err() {
        return Err("burn-onnx failed while generating the decoder burnpack".into());
    }

    Ok(())
}

fn float_tensor<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &BurnDevice,
) -> Tensor<BurnBackend, D> {
    Tensor::from_data(TensorData::new(values, shape).convert::<f32>(), device)
}

fn int_tensor<const D: usize>(
    values: Vec<i64>,
    shape: [usize; D],
    device: &BurnDevice,
) -> Tensor<BurnBackend, D, Int> {
    Tensor::from_data(TensorData::new(values, shape).convert::<i64>(), device)
}

fn array3_tensor(array: &Array3<f32>, device: &BurnDevice) -> Tensor<BurnBackend, 3> {
    let shape = [array.shape()[0], array.shape()[1], array.shape()[2]];
    float_tensor(array.iter().copied().collect(), shape, device)
}

fn array4_tensor(array: &Array4<f32>, device: &BurnDevice) -> Tensor<BurnBackend, 4> {
    let shape = [
        array.shape()[0],
        array.shape()[1],
        array.shape()[2],
        array.shape()[3],
    ];
    float_tensor(array.iter().copied().collect(), shape, device)
}

fn tensor_to_array3(tensor: Tensor<BurnBackend, 3>) -> Result<Array3<f32>, Box<dyn Error>> {
    let data = tensor.to_data();
    let shape = (
        data.shape[0] as usize,
        data.shape[1] as usize,
        data.shape[2] as usize,
    );
    Ok(Array3::from_shape_vec(shape, data.to_vec::<f32>()?)?)
}

fn tensor_to_array4(tensor: Tensor<BurnBackend, 4>) -> Result<Array4<f32>, Box<dyn Error>> {
    let data = tensor.to_data();
    let shape = (
        data.shape[0] as usize,
        data.shape[1] as usize,
        data.shape[2] as usize,
        data.shape[3] as usize,
    );
    Ok(Array4::from_shape_vec(shape, data.to_vec::<f32>()?)?)
}
