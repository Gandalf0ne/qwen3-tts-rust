use crate::assets_manager::Assets;
use crate::models::llama::{LlamaBatch, LlamaContext, LlamaModel, LlamaSampler};
use crate::models::onnx::{init_onruntime, AudioDecoder, AudioEncoder, SpeakerEncoder};
use crate::tts::prompt::PromptBuilder;
use crate::utils::cache;
use crate::utils::tokenizer::Tokenizer;
use crate::utils::voice_file::VoiceFile;
use crate::AudioSample;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Sampler configuration for TTS generation
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for sampling (higher = more random, 0.0 = greedy)
    pub temperature: f32,
    /// Top-K sampling (0 = disabled)
    pub top_k: i32,
    /// Top-P (nucleus) sampling (1.0 = disabled)
    pub top_p: f32,
    /// Min-P sampling threshold (0.0 = disabled)
    pub min_p: f32,
    /// Repeat penalty (1.0 = disabled)
    pub repeat_penalty: f32,
    /// Frequency penalty (0.0 = disabled)
    pub frequency_penalty: f32,
    /// Presence penalty (0.0 = disabled)
    pub presence_penalty: f32,
    /// Number of recent tokens to consider for penalties
    pub penalty_last_n: usize,
    /// Random seed (None = use system entropy)
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
            seed: None,
        }
    }
}

impl SamplerConfig {
    pub fn new(temperature: f32, top_k: i32, top_p: f32, seed: Option<u64>) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            repeat_penalty: 1.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
            seed,
        }
    }

    pub fn with_penalties(
        mut self,
        min_p: f32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        penalty_last_n: usize,
    ) -> Self {
        self.min_p = min_p;
        self.repeat_penalty = repeat_penalty;
        self.frequency_penalty = frequency_penalty;
        self.presence_penalty = presence_penalty;
        self.penalty_last_n = penalty_last_n;
        self
    }
}

#[derive(Clone, Debug)]
struct AmdGpuMonitor {
    label: String,
    gpu_busy_path: PathBuf,
    mem_busy_path: Option<PathBuf>,
}

#[derive(Debug)]
struct AmdGpuSnapshot {
    gpu_busy_percent: u32,
    mem_busy_percent: Option<u32>,
}

#[derive(Debug, Default)]
struct DecodeProgressStats {
    device_label: Option<String>,
    gpu_busy_samples: usize,
    gpu_busy_sum: u64,
    gpu_busy_max: Option<u32>,
    mem_busy_samples: usize,
    mem_busy_sum: u64,
    mem_busy_max: Option<u32>,
}

struct DecodeProgressLogger {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<DecodeProgressStats>>,
}

impl AmdGpuMonitor {
    fn detect() -> Option<Self> {
        if !cfg!(target_os = "linux") {
            return None;
        }

        let drm_entries = std::fs::read_dir("/sys/class/drm").ok()?;

        for entry in drm_entries.filter_map(Result::ok) {
            let entry_name = entry.file_name();
            let entry_name = entry_name.to_str()?;

            if !entry_name.starts_with("card") || entry_name.contains('-') {
                continue;
            }

            let device_dir = entry.path().join("device");
            let vendor = std::fs::read_to_string(device_dir.join("vendor")).ok()?;
            if vendor.trim() != "0x1002" {
                continue;
            }

            let gpu_busy_path = device_dir.join("gpu_busy_percent");
            if !gpu_busy_path.is_file() {
                continue;
            }

            let mem_busy_candidate = device_dir.join("mem_busy_percent");
            let label = std::fs::read_to_string(device_dir.join("uevent"))
                .ok()
                .and_then(|uevent| {
                    uevent.lines().find_map(|line| {
                        line.strip_prefix("PCI_SLOT_NAME=")
                            .map(|value| format!("{} ({})", entry_name, value))
                    })
                })
                .unwrap_or_else(|| entry_name.to_string());

            return Some(Self {
                label,
                gpu_busy_path,
                mem_busy_path: mem_busy_candidate.is_file().then_some(mem_busy_candidate),
            });
        }

        None
    }

    fn read_snapshot(&self) -> Option<AmdGpuSnapshot> {
        let gpu_busy_percent = Self::read_percent(&self.gpu_busy_path)?;
        let mem_busy_percent = self
            .mem_busy_path
            .as_ref()
            .and_then(|path| Self::read_percent(path));

        Some(AmdGpuSnapshot {
            gpu_busy_percent,
            mem_busy_percent,
        })
    }

    fn read_percent(path: &Path) -> Option<u32> {
        std::fs::read_to_string(path).ok()?.trim().parse::<u32>().ok()
    }
}

impl DecodeProgressStats {
    fn from_monitor(monitor: &Option<AmdGpuMonitor>) -> Self {
        Self {
            device_label: monitor.as_ref().map(|monitor| monitor.label.clone()),
            ..Self::default()
        }
    }

    fn record(&mut self, snapshot: &AmdGpuSnapshot) {
        self.gpu_busy_samples += 1;
        self.gpu_busy_sum += snapshot.gpu_busy_percent as u64;
        self.gpu_busy_max = Some(
            self.gpu_busy_max
                .map(|current| current.max(snapshot.gpu_busy_percent))
                .unwrap_or(snapshot.gpu_busy_percent),
        );

        if let Some(mem_busy_percent) = snapshot.mem_busy_percent {
            self.mem_busy_samples += 1;
            self.mem_busy_sum += mem_busy_percent as u64;
            self.mem_busy_max = Some(
                self.mem_busy_max
                    .map(|current| current.max(mem_busy_percent))
                    .unwrap_or(mem_busy_percent),
            );
        }
    }

    fn merge(&mut self, other: Self) {
        if self.device_label.is_none() {
            self.device_label = other.device_label;
        }

        self.gpu_busy_samples += other.gpu_busy_samples;
        self.gpu_busy_sum += other.gpu_busy_sum;
        self.gpu_busy_max = match (self.gpu_busy_max, other.gpu_busy_max) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };

        self.mem_busy_samples += other.mem_busy_samples;
        self.mem_busy_sum += other.mem_busy_sum;
        self.mem_busy_max = match (self.mem_busy_max, other.mem_busy_max) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
    }

    fn gpu_busy_avg(&self) -> Option<f64> {
        (self.gpu_busy_samples > 0)
            .then(|| self.gpu_busy_sum as f64 / self.gpu_busy_samples as f64)
    }

    fn mem_busy_avg(&self) -> Option<f64> {
        (self.mem_busy_samples > 0).then(|| self.mem_busy_sum as f64 / self.mem_busy_samples as f64)
    }

    fn summary_suffix(&self) -> String {
        let mut parts = Vec::new();

        if let Some(device_label) = &self.device_label {
            parts.push(format!("gpu_device={}", device_label));
        }

        if let Some(gpu_busy_avg) = self.gpu_busy_avg() {
            parts.push(format!("gpu_busy_avg={:.1}%", gpu_busy_avg));
        }

        if let Some(gpu_busy_max) = self.gpu_busy_max {
            parts.push(format!("gpu_busy_max={}%", gpu_busy_max));
        }

        if let Some(mem_busy_avg) = self.mem_busy_avg() {
            parts.push(format!("mem_busy_avg={:.1}%", mem_busy_avg));
        }

        if let Some(mem_busy_max) = self.mem_busy_max {
            parts.push(format!("mem_busy_max={}%", mem_busy_max));
        }

        if parts.is_empty() {
            String::new()
        } else {
            format!(" {}", parts.join(" "))
        }
    }
}

impl DecodeProgressLogger {
    fn spawn(
        phase_label: String,
        monitor: Option<AmdGpuMonitor>,
    ) -> Self {
        let interval_ms = std::env::var("QWEN3_TTS_DECODE_PROGRESS_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1000);

        if interval_ms == 0 {
            return Self {
                stop: Arc::new(AtomicBool::new(true)),
                handle: None,
            };
        }

        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = stop.clone();
        let interval = Duration::from_millis(interval_ms);

        let handle = std::thread::spawn(move || {
            let started = Instant::now();
            let mut stats = DecodeProgressStats::from_monitor(&monitor);

            loop {
                std::thread::sleep(interval);

                if stop_worker.load(Ordering::Relaxed) {
                    break;
                }

                let elapsed = started.elapsed().as_secs_f64();
                match monitor.as_ref().and_then(|monitor| monitor.read_snapshot()) {
                    Some(snapshot) => {
                        stats.record(&snapshot);
                        match snapshot.mem_busy_percent {
                            Some(mem_busy_percent) => println!(
                                "{}: elapsed={:.2}s gpu_busy={}%% mem_busy={}%%",
                                phase_label, elapsed, snapshot.gpu_busy_percent, mem_busy_percent
                            ),
                            None => println!(
                                "{}: elapsed={:.2}s gpu_busy={}%%",
                                phase_label, elapsed, snapshot.gpu_busy_percent
                            ),
                        }
                    }
                    None => println!("{}: elapsed={:.2}s", phase_label, elapsed),
                }
            }

            stats
        });

        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> DecodeProgressStats {
        self.stop.store(true, Ordering::Relaxed);
        match self.handle.take() {
            Some(handle) => handle.join().unwrap_or_default(),
            None => DecodeProgressStats::default(),
        }
    }
}

/// Main TTS Engine Struct
///
/// IMPORTANT: Field ordering matters for Drop!
/// Rust drops fields in declaration order. Contexts MUST be declared before models
/// because context destructors reference model memory. If models are dropped first,
/// context destructors will access freed memory (ACCESS_VIOLATION).
pub struct TtsEngine {
    assets: Assets,
    tokenizer: Tokenizer,
    // ONNX Models
    encoder: Option<AudioEncoder>,
    speaker_encoder: Option<SpeakerEncoder>,
    decoder: Arc<Mutex<AudioDecoder>>,
    // Llama: Contexts MUST be listed before models for correct drop order
    talker_ctx: LlamaContext,
    predictor_ctx: LlamaContext,
    talker_model: LlamaModel,
    predictor_model: LlamaModel,

    // Speakers Cache
    speakers: HashMap<String, VoiceFile>,

    // Config
    _model_dir: PathBuf,
    max_steps: usize,
    sampler_config: SamplerConfig,
}

impl TtsEngine {
    fn has_linux_dri_device() -> bool {
        if !cfg!(target_os = "linux") {
            return false;
        }

        std::fs::read_dir("/dev/dri")
            .ok()
            .map(|entries| {
                entries.filter_map(Result::ok).any(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .map(|name| name.starts_with("renderD") || name.starts_with("card"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    }

    fn runtime_backend_override() -> Option<String> {
        std::env::var("QWEN3_TTS_RUNTIME_BACKEND")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
    }

    fn allow_llama_cpu_fallback() -> bool {
        std::env::var("QWEN3_TTS_LLAMA_CPU_FALLBACK").unwrap_or_default() == "1"
    }

    fn n_gpu_layers_from_env() -> i32 {
        if let Ok(value) = std::env::var("QWEN3_TTS_N_GPU_LAYERS") {
            if let Ok(parsed) = value.trim().parse::<i32>() {
                return parsed;
            }
            eprintln!(
                "Invalid QWEN3_TTS_N_GPU_LAYERS value '{}', falling back to automatic default",
                value
            );
        }

        match Self::runtime_backend_override().as_deref() {
            Some("cpu") => 0,
            None if cfg!(target_os = "linux") && !Self::has_linux_dri_device() => 0,
            _ => 99,
        }
    }

    fn load_llama_models(
        talker_path: &Path,
        predictor_path: &Path,
        n_gpu_layers: i32,
    ) -> Result<(LlamaModel, LlamaModel), String> {
        let talker_model = LlamaModel::load(talker_path, n_gpu_layers)
            .map_err(|e| format!("Failed to load Talker: {}", e))?;

        let predictor_model = match LlamaModel::load(predictor_path, n_gpu_layers) {
            Ok(model) => model,
            Err(e) => {
                drop(talker_model);
                return Err(format!("Failed to load Predictor: {}", e));
            }
        };

        Ok((talker_model, predictor_model))
    }

    /// Initialize the TTS Engine from the specified model directory.
    ///
    /// This function loads all necessary models (GGUF, Onnx, Tokenizer) from the given directory.
    /// It ensures that the essential components for inference are present.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to the directory containing model files.
    /// * `quant` - Quantization level (e.g., "none", "q5_k_m", "q8_0").
    /// * `n_threads` - Number of threads to use for generation (default: 4 if <= 0).
    pub async fn new(
        model_dir: impl AsRef<Path>,
        quant: &str,
        _n_threads: i32,
    ) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        println!("Loading TtsEngine from: {:?} (quant: {})", model_dir, quant);
        let n_gpu_layers = Self::n_gpu_layers_from_env();
        println!("GPU layers: {}", n_gpu_layers);

        // 0. Auto-download check (Models + Runtimes)
        Self::download_models(model_dir, quant).await?;

        let quant_dir = match quant {
            "q5_k_m" => "gguf_q5_k_m",
            "q8_0" => "gguf_q8_0",
            _ => "gguf",
        };

        // 1. Assets - 总是从未量化的 gguf 目录加载（assets 不应该被量化）
        let assets_path = model_dir.join("gguf");
        let assets =
            Assets::load(&assets_path).map_err(|e| format!("Failed to load assets: {}", e))?;

        // 2. Tokenizer
        let tokenizer =
            Tokenizer::load(model_dir).map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // 3. Initialize ONNX Runtime (must be called before any ONNX session)
        println!("Initializing ONNX Runtime...");
        init_onruntime().map_err(|e| format!("Failed to init ONNX Runtime: {}", e))?;

        // 4. ONNX Models (Optional for preset mode, but good to have)
        let onnx_dir = model_dir.join("onnx");
        let encoder_path = onnx_dir.join("qwen3_tts_codec_encoder.onnx");
        let encoder = match AudioEncoder::load(&encoder_path.to_string_lossy()) {
            Ok(encoder) => Some(encoder),
            Err(error) => {
                eprintln!("Failed to load AudioEncoder ({}): {}", encoder_path.display(), error);
                None
            }
        };

        let speaker_encoder_path = onnx_dir.join("qwen3_tts_speaker_encoder.onnx");
        let speaker_encoder = match SpeakerEncoder::load(&speaker_encoder_path.to_string_lossy()) {
            Ok(encoder) => Some(encoder),
            Err(error) => {
                eprintln!(
                    "Failed to load SpeakerEncoder ({}): {}",
                    speaker_encoder_path.display(),
                    error
                );
                None
            }
        };

        // 5. Load GGUF Models
        let talker_path = model_dir.join(quant_dir).join("qwen3_tts_talker.gguf");
        let predictor_path = model_dir.join(quant_dir).join("qwen3_tts_predictor.gguf");

        let (talker_model, predictor_model, effective_gpu_layers) =
            match Self::load_llama_models(&talker_path, &predictor_path, n_gpu_layers) {
                Ok((talker_model, predictor_model)) => {
                    (talker_model, predictor_model, n_gpu_layers)
                }
                Err(primary_error) if n_gpu_layers > 0 && Self::allow_llama_cpu_fallback() => {
                    eprintln!(
                        "GPU/Vulkan model initialization failed ({}). Retrying on CPU because QWEN3_TTS_LLAMA_CPU_FALLBACK=1.",
                        primary_error
                    );

                    let (talker_model, predictor_model) =
                        Self::load_llama_models(&talker_path, &predictor_path, 0).map_err(
                            |cpu_error| {
                                format!(
                                    "{}; CPU fallback also failed: {}",
                                    primary_error, cpu_error
                                )
                            },
                        )?;

                    println!("GPU fallback active. Continuing with CPU layers.");
                    (talker_model, predictor_model, 0)
                }
                Err(error) => return Err(error),
            };

        if effective_gpu_layers != n_gpu_layers {
            println!("Effective GPU layers: {}", effective_gpu_layers);
        }

        // 5. Create Contexts
        // talker: n_ctx=4096, n_batch=2048, embeddings=1, threads=-1 (auto)
        let talker_ctx = LlamaContext::new(&talker_model, 4096, 2048, 1, -1)
            .map_err(|e| format!("Failed to create Talker context: {}", e))?;

        // predictor: n_ctx=512, n_batch=32, embeddings=0, threads=4
        let predictor_ctx = LlamaContext::new(&predictor_model, 512, 32, 0, 4)
            .map_err(|e| format!("Failed to create Predictor context: {}", e))?;

        // 6. 预加载 decoder（预热）
        println!("Pre-loading AudioDecoder...");
        let decoder =
            AudioDecoder::load(&onnx_dir.join("qwen3_tts_decoder.onnx").to_string_lossy())
                .map_err(|e| format!("Failed to load AudioDecoder: {}", e))?;
        let decoder = Arc::new(Mutex::new(decoder));
        println!("AudioDecoder pre-loaded and warmed up.");

        println!("TtsEngine loaded successfully.");

        let mut engine = Self {
            assets,
            tokenizer,
            encoder,
            speaker_encoder,
            decoder,
            talker_model,
            predictor_model,
            talker_ctx,
            predictor_ctx,
            speakers: HashMap::new(),
            _model_dir: model_dir.to_path_buf(),
            max_steps: 4096,
            sampler_config: SamplerConfig::default(),
        };

        // 6. Load Speakers
        let speakers_dir = model_dir.join("preset_speakers"); // Default to preset directory
        let speakers_dir = if speakers_dir.exists() {
            speakers_dir
        } else {
            PathBuf::from("speakers")
        };

        if speakers_dir.exists() {
            engine.load_speakers(&speakers_dir)?;
        }

        Ok(engine)
    }

    /// Set the maximum number of generation steps (tokens).
    pub fn set_max_steps(&mut self, steps: usize) {
        self.max_steps = steps;
    }

    /// Set the sampler configuration for generation.
    pub fn set_sampler_config(&mut self, config: SamplerConfig) {
        self.sampler_config = config;
    }

    /// Get the current sampler configuration.
    pub fn get_sampler_config(&self) -> &SamplerConfig {
        &self.sampler_config
    }

    /// Get mutable sampler configuration.
    pub fn get_sampler_config_mut(&mut self) -> &mut SamplerConfig {
        &mut self.sampler_config
    }

    /// Get the speakers map.
    pub fn get_speakers_map(&self) -> &HashMap<String, VoiceFile> {
        &self.speakers
    }

    /// Load all speakers from the specified directory.
    pub fn load_speakers(&mut self, speakers_dir: impl AsRef<Path>) -> Result<(), String> {
        let speakers_dir = speakers_dir.as_ref();
        println!("Loading speakers from: {:?}", speakers_dir);

        let entries = std::fs::read_dir(speakers_dir).map_err(|e| e.to_string())?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(voice) = VoiceFile::load(&path) {
                    let id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    self.speakers.insert(id, voice);
                }
            }
        }
        println!("Loaded {} speakers.", self.speakers.len());
        Ok(())
    }

    /// Get a speaker by ID or name, with fallback to "vivian".
    pub fn get_speaker(&self, id_or_name: &str) -> &VoiceFile {
        if let Some(v) = self.speakers.get(id_or_name) {
            return v;
        }
        // Fallback to name match
        for v in self.speakers.values() {
            if let Some(ref name) = v.name {
                if name == id_or_name {
                    return v;
                }
            }
        }
        // Final fallback to vivian
        self.speakers.get("vivian").unwrap_or_else(|| {
            // Panic if even vivian is missing and cache is empty
            self.speakers
                .values()
                .next()
                .expect("No speakers loaded in engine!")
        })
    }

    /// Helper to download necessary files before loading.
    pub async fn download_models(model_dir: impl AsRef<Path>, quant: &str) -> Result<(), String> {
        let downloader = crate::download::Downloader::new().await;
        downloader
            .check_and_download(model_dir.as_ref(), quant)
            .await
            .map_err(|e| format!("Download failed: {}", e))
    }

    /// Generate speech from text using a reference audio.
    pub fn generate(
        &mut self,
        text: &str,
        ref_audio_path: impl AsRef<Path>,
        ref_text: &str,
        instruct: Option<&str>,
    ) -> Result<AudioSample, String> {
        let ref_audio_path = ref_audio_path.as_ref();

        // 1. Process Reference Audio
        let (ref_codes, spk_emb) = self.process_reference(ref_audio_path)?;

        // 2. Build Prompt
        // lang_id = 2055 (Chinese) hardcoded for now or parameterize later
        let ref_text_ids = self.tokenizer.encode(ref_text);
        let ref_codes_i32: Vec<i32> = ref_codes.iter().map(|&c| c as i32).collect();

        let prompt_data = PromptBuilder::build_clone_prompt(
            text,
            &self.tokenizer,
            &self.assets,
            &ref_codes_i32,
            &ref_text_ids,
            &spk_emb,
            2055,
            instruct,
        );

        self.run_inference(prompt_data)
    }

    /// Process reference audio to get codes and speaker embedding, using cache if available.
    fn process_reference(&mut self, audio_path: &Path) -> Result<(Vec<i64>, Vec<f32>), String> {
        let cache_path = audio_path.with_extension("cache");
        if cache_path.exists() {
            if let Ok((c, e)) = cache::load_cache(&cache_path) {
                return Ok((c, e));
            }
        }

        let audio = AudioSample::load_wav(audio_path)
            .map_err(|e| format!("Failed to load audio: {}", e))?;

        let ref_codes = self
            .encoder
            .as_mut()
            .ok_or("AudioEncoder not loaded (required for processing raw audio)".to_string())?
            .encode(&audio.samples)
            .map_err(|e| format!("Audio encode failed: {}", e))?;
        let spk_emb = self
            .speaker_encoder
            .as_mut()
            .ok_or("SpeakerEncoder not loaded (required for processing raw audio)".to_string())?
            .encode(&audio.samples)
            .map_err(|e| format!("Speaker extraction failed: {}", e))?;

        let _ = cache::save_cache(&cache_path, &ref_codes, &spk_emb);

        Ok((ref_codes, spk_emb))
    }

    // --- Helpers ---

    fn qwen3_position(start: i32, len: usize) -> Vec<i32> {
        let mut pos = Vec::with_capacity(len * 4);
        let range: Vec<i32> = (start..start + len as i32).collect();
        pos.extend_from_slice(&range); // Temporal
        pos.extend_from_slice(&range); // Height
        pos.extend_from_slice(&range); // Width
        pos.extend(std::iter::repeat_n(0, len)); // Channel
        pos
    }

    fn normal_position(cur_pos: usize, n_tokens: usize) -> Vec<i32> {
        (0..n_tokens).map(|i| (cur_pos + i) as i32).collect()
    }

    /// Create a VoiceFile from a reference audio file and its text.
    ///
    /// Requires that AudioEncoder and SpeakerEncoder are loaded.
    /// The reference audio MUST be 24000Hz.
    pub fn create_voice_file(
        &mut self,
        audio_path: impl AsRef<Path>,
        ref_text: String,
    ) -> Result<crate::utils::voice_file::VoiceFile, String> {
        let encoder = self.encoder.as_mut().ok_or(
            "AudioEncoder not loaded. Please ensure models/onnx/qwen3_tts_codec_encoder.onnx exists.",
        )?;
        let speaker_encoder = self.speaker_encoder.as_mut().ok_or(
            "SpeakerEncoder not loaded. Please ensure models/onnx/qwen3_tts_speaker_encoder.onnx exists.",
        )?;

        // 1. Load Audio
        let mut reader =
            hound::WavReader::open(audio_path).map_err(|e| format!("WAV error: {}", e))?;
        let spec = reader.spec();

        if spec.sample_rate != 24000 {
            return Err(format!(
                "Expected 24000Hz audio, found {}Hz",
                spec.sample_rate
            ));
        }

        let audio: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => {
                reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
            }
            (hound::SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .map(|s| (s.unwrap_or(0) as f32) / 32768.0)
                .collect(),
            (hound::SampleFormat::Int, 32) => reader
                .samples::<i32>()
                .map(|s| (s.unwrap_or(0) as f32) / 2147483648.0)
                .collect(),
            _ => {
                return Err(format!(
                    "Unsupported WAV format: {:?} {} bits",
                    spec.sample_format, spec.bits_per_sample
                ))
            }
        };

        // If stereo, take channel 1
        let audio = if spec.channels > 1 {
            audio.chunks(spec.channels as usize).map(|c| c[0]).collect()
        } else {
            audio
        };

        // 2. Run Encoders
        println!("Extracting audio codes...");
        let audio_codes = encoder.encode(&audio).map_err(|e| e.to_string())?;

        println!("Extracting speaker embedding...");
        let speaker_embedding = speaker_encoder.encode(&audio).map_err(|e| e.to_string())?;

        Ok(crate::utils::voice_file::VoiceFile::new(
            ref_text,
            audio_codes,
            speaker_embedding,
        ))
    }

    /// Generate speech using a pre-loaded VoiceFile.
    pub fn generate_with_voice(
        &mut self,
        text: &str,
        voice: &crate::VoiceFile,
        instruct: Option<&str>,
    ) -> Result<AudioSample, String> {
        self.generate_with_voice_streaming(text, voice, instruct, None)
    }

    /// Generate speech using a pre-loaded VoiceFile.
    pub fn generate_with_voice_streaming(
        &mut self,
        text: &str,
        voice: &crate::VoiceFile,
        instruct: Option<&str>,
        stream_tx: Option<tokio::sync::mpsc::UnboundedSender<Vec<f32>>>,
    ) -> Result<AudioSample, String> {
        let prompt_data = if voice.audio_codes.is_empty() {
            PromptBuilder::build_core(
                text,
                &self.tokenizer,
                &self.assets,
                Some(2055),
                None,
                Some(&voice.speaker_embedding),
                instruct,
                None,
            )
        } else {
            let ref_text_ids = self.tokenizer.encode(&voice.ref_text);
            let ref_codes_i32: Vec<i32> = voice.audio_codes.iter().map(|&c| c as i32).collect();

            Ok(PromptBuilder::build_clone_prompt(
                text,
                &self.tokenizer,
                &self.assets,
                &ref_codes_i32,
                &ref_text_ids,
                &voice.speaker_embedding,
                2055,
                instruct,
            ))
        }?;

        self.run_inference_stream(prompt_data, stream_tx)
    }

    fn run_inference(
        &mut self,
        prompt_data: crate::tts::prompt::PromptData,
    ) -> Result<AudioSample, String> {
        self.run_inference_stream(prompt_data, None)
    }

    fn run_inference_stream(
        &mut self,
        prompt_data: crate::tts::prompt::PromptData,
        stream_tx: Option<tokio::sync::mpsc::UnboundedSender<Vec<f32>>>,
    ) -> Result<AudioSample, String> {
        let generation_started = std::time::Instant::now();
        self.talker_ctx.clear_kv_cache();
        self.predictor_ctx.clear_kv_cache();

        let n_tokens_prompt = prompt_data.embd.len();
        let prompt_embeds_flat: Vec<f32> = prompt_data.embd.iter().flatten().copied().collect();
        let talker_embd = self.talker_model.n_embd;
        let predictor_embd = self.predictor_model.n_embd;

        let mut talker_batch = LlamaBatch::new(4096, talker_embd, 1, 4);
        let pos_arr = Self::qwen3_position(0, n_tokens_prompt);
        talker_batch.set_embd(&prompt_embeds_flat, &pos_arr, 0);

        self.talker_ctx
            .decode(&mut talker_batch)
            .map_err(|e| format!("Talker prefill failed: {}", e))?;

        let mut all_codes: Vec<i32> = Vec::new();
        let mut cur_pos = n_tokens_prompt;

        let mut predictor_batch = LlamaBatch::new(32, predictor_embd, 1, 1);
        let predictor_sampler = LlamaSampler::greedy(self.predictor_model.n_vocab);

        let seed = self.sampler_config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or_else(|_| rand::random())
        });
        let talker_sampler = LlamaSampler::new(
            self.talker_model.n_vocab,
            self.sampler_config.temperature,
            self.sampler_config.top_k,
            self.sampler_config.top_p,
            seed,
        );

        let streaming_decode_enabled = stream_tx.is_some();
        let (tx, rx) = std::sync::mpsc::channel::<(Vec<i64>, bool)>();
        let decoder_arc = self.decoder.clone();
        let decode_gpu_monitor = if streaming_decode_enabled {
            None
        } else {
            AmdGpuMonitor::detect()
        };
        let stream_decode_interval = std::env::var("QWEN3_TTS_STREAM_DECODE_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(4);
        let non_stream_decode_interval = std::env::var("QWEN3_TTS_NONSTREAM_DECODE_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(32);

        struct DecoderThreadResult {
            audio: Vec<f32>,
            active_decode_time: std::time::Duration,
            chunk_count: usize,
            frame_count: usize,
            progress_stats: DecodeProgressStats,
        }

        let streaming_decode_enabled_for_decoder = streaming_decode_enabled;
        let decode_gpu_monitor_for_decoder = decode_gpu_monitor.clone();
        let decoder_handle = std::thread::spawn(move || {
            let mut full_audio = Vec::new();
            let mut state = {
                let local_decoder = decoder_arc.lock().unwrap();
                local_decoder.create_state()
            };
            let mut active_decode_time = std::time::Duration::default();
            let mut chunk_count = 0usize;
            let mut frame_count = 0usize;
            let mut progress_stats =
                DecodeProgressStats::from_monitor(&decode_gpu_monitor_for_decoder);
            let mut decode_job_index = 0usize;

            while let Ok((codes, is_final)) = rx.recv() {
                let n_frames = codes.len() / 16;

                if n_frames == 0 {
                    if is_final {
                        let mut local_decoder = decoder_arc.lock().unwrap();
                        if let Ok(samples) = local_decoder.decode(&[], &mut state, true) {
                            if !samples.is_empty() {
                                if let Some(ref stx) = stream_tx {
                                    let _ = stx.send(samples.clone());
                                }
                                full_audio.extend(samples);
                            }
                        }
                        break;
                    }
                    continue;
                }

                let safe_codes: Vec<i64> = codes.iter().map(|&c| c.clamp(0, 2047)).collect();

                decode_job_index += 1;

                let progress_logger = if !streaming_decode_enabled_for_decoder {
                    match decode_gpu_monitor_for_decoder.as_ref() {
                        Some(monitor) => println!(
                            "Post-EOS decode chunk {} started: frames={} final={} gpu_device={}",
                            decode_job_index, n_frames, is_final, monitor.label
                        ),
                        None => println!(
                            "Post-EOS decode chunk {} started: frames={} final={} gpu_telemetry=unavailable",
                            decode_job_index, n_frames, is_final
                        ),
                    }

                    Some(DecodeProgressLogger::spawn(
                        format!("Post-EOS decode progress [chunk {}]", decode_job_index),
                        decode_gpu_monitor_for_decoder.clone(),
                    ))
                } else {
                    None
                };

                let chunk_started = std::time::Instant::now();
                let mut local_decoder = decoder_arc.lock().unwrap();
                let decode_result = local_decoder.decode(&safe_codes, &mut state, is_final);
                drop(local_decoder);

                let chunk_elapsed = chunk_started.elapsed();
                active_decode_time += chunk_elapsed;

                let chunk_progress_stats = progress_logger
                    .map(|logger| logger.finish())
                    .unwrap_or_default();
                progress_stats.merge(chunk_progress_stats);

                match decode_result {
                    Ok(samples) => {
                        chunk_count += 1;
                        frame_count += n_frames;

                        if !streaming_decode_enabled_for_decoder {
                            println!(
                                "Post-EOS decode chunk {} finished: elapsed={:.2}s frames={} samples={}{}",
                                decode_job_index,
                                chunk_elapsed.as_secs_f64(),
                                n_frames,
                                samples.len(),
                                progress_stats.summary_suffix()
                            );
                        }

                        if !samples.is_empty() {
                            if let Some(ref stx) = stream_tx {
                                let _ = stx.send(samples.clone());
                            }
                            full_audio.extend(samples);
                        }
                    }
                    Err(error) => {
                        println!(
                            "Post-EOS decode chunk {} failed after {:.2}s: {}",
                            decode_job_index,
                            chunk_elapsed.as_secs_f64(),
                            error
                        );
                    }
                }
            }
            DecoderThreadResult {
                audio: full_audio,
                active_decode_time,
                chunk_count,
                frame_count,
                progress_stats,
            }
        });

        let tts_pad = self.assets.tts_pad.clone();

        // 连续静音检测参数 (12.5Hz = 80ms/帧)
        const MAX_SILENT_FRAMES: usize = 15; // 1.2秒强制停止
        const SILENT_PENALTY_THRESHOLD: usize = 4; // 0.3秒(4帧)后开始惩罚
        const SILENT_PENALTY_MAX_FRAMES: usize = 8; // 0.6秒(8帧)后极大惩罚
        const SILENT_PENALTY_VALUE: f32 = 2.0;

        let mut consecutive_silent_frames: usize = 0;
        let mut last_sent_frame: usize = 0;

        for step in 0..self.max_steps {
            print!("\r    Generation Step {}/{}...", step + 1, self.max_steps);
            let _ = std::io::Write::flush(&mut std::io::stdout());

            // Talker
            let sample_idx = if cur_pos == n_tokens_prompt {
                (n_tokens_prompt - 1) as i32
            } else {
                0
            };

            let allow_tokens: Vec<i32> = vec![2150, 2148, 2149];

            let code_0 = if consecutive_silent_frames >= SILENT_PENALTY_THRESHOLD {
                let penalty = if consecutive_silent_frames >= SILENT_PENALTY_MAX_FRAMES {
                    100.0 // 极大惩罚，强制退出静音
                } else {
                    SILENT_PENALTY_VALUE
                        * (1.0
                            + (consecutive_silent_frames - SILENT_PENALTY_THRESHOLD) as f32 * 0.5)
                };
                eprintln!(
                    "[Debug] silent {} frames, penalty: {:.1}",
                    consecutive_silent_frames, penalty
                );
                talker_sampler.sample_with_silent_penalty(
                    &self.talker_ctx,
                    sample_idx,
                    Some(0),
                    Some(2048),
                    Some(&allow_tokens),
                    penalty,
                    SILENT_PENALTY_MAX_FRAMES, // 只对前8帧应用惩罚
                )
            } else {
                talker_sampler.sample_with_allow(
                    &self.talker_ctx,
                    sample_idx,
                    Some(0),
                    Some(2048),
                    Some(&allow_tokens),
                )
            };
            eprintln!("[Debug] Sampled code_0={}", code_0);

            if code_0 == 2150 || code_0 == 151673 {
                println!("\n    EOS detected at step {} (code_0={})", step, code_0);
                break;
            }

            let is_silent_frame = code_0 < 100;
            if is_silent_frame {
                consecutive_silent_frames += 1;
                eprintln!(
                    "[Debug] silent frame {}, code_0={}",
                    consecutive_silent_frames, code_0
                );
                if consecutive_silent_frames >= MAX_SILENT_FRAMES {
                    eprintln!(
                        "[Debug] Early stop: {} consecutive silent frames",
                        consecutive_silent_frames
                    );
                    println!(
                        "\n    Early stop: {} consecutive silent frames detected",
                        consecutive_silent_frames
                    );
                    break;
                }
            } else {
                consecutive_silent_frames = 0;
            }

            all_codes.push(code_0);

            // Predictor
            let emb_idx = if step == 0 { n_tokens_prompt - 1 } else { 0 };
            let m_hidden = self.talker_ctx.get_embedding_at(emb_idx).to_vec();

            let m_h_1024 = self.assets.project(&m_hidden);
            let code_0_1024 = self.assets.get_codec_embedding_1024(0, code_0);

            let mut predictor_input = Vec::with_capacity(2 * predictor_embd);
            predictor_input.extend_from_slice(&m_h_1024);
            predictor_input.extend_from_slice(&code_0_1024);

            self.predictor_ctx.clear_kv_cache();
            predictor_batch.clear();
            let pred_pos = Self::normal_position(0, 2);
            predictor_batch.set_embd(&predictor_input, &pred_pos, 0);

            self.predictor_ctx
                .decode(&mut predictor_batch)
                .map_err(|e| format!("Predictor prefill failed: {}", e))?;

            let mut step_embeds_2048: Vec<Vec<f32>> = Vec::new();
            step_embeds_2048.push(self.assets.get_codec_embedding(0, code_0));

            for q in 1..16 {
                let start_offset = (q - 1) * 2048;
                let end_offset = q * 2048;
                let sampled = predictor_sampler.sample(
                    &self.predictor_ctx,
                    0,
                    Some(start_offset),
                    Some(end_offset),
                );
                let code_q = sampled - start_offset as i32;
                all_codes.push(code_q);

                let emb = self.assets.get_codec_embedding(q, code_q);
                step_embeds_2048.push(emb.to_vec());

                if q < 15 {
                    let next_embed_1024 = self.assets.get_codec_embedding_1024(q, code_q);
                    let next_pos = Self::normal_position(q + 1, 1);
                    predictor_batch.clear();
                    predictor_batch.set_embd(&next_embed_1024, &next_pos, 0);
                    self.predictor_ctx
                        .decode(&mut predictor_batch)
                        .map_err(|e| format!("Predictor decode failed: {}", e))?;
                }
            }

            // 流式发送
            let current_frame = all_codes.len() / 16;

            if streaming_decode_enabled && current_frame >= last_sent_frame + stream_decode_interval {
                let start_frame = last_sent_frame;
                let end_frame = last_sent_frame + stream_decode_interval;

                let start_idx = start_frame * 16;
                let end_idx = end_frame * 16;
                let frame_codes: Vec<i64> = all_codes[start_idx..end_idx]
                    .iter()
                    .map(|&c| c as i64)
                    .collect();

                let _ = tx.send((frame_codes, false));
                last_sent_frame += stream_decode_interval;
            }

            let mut feedback = vec![0.0f32; 2048];
            for embed in &step_embeds_2048 {
                for (i, val) in embed.iter().enumerate() {
                    feedback[i] += val;
                }
            }

            let text_vec = &tts_pad;
            for (i, val) in text_vec.iter().enumerate() {
                if i < feedback.len() {
                    feedback[i] += val;
                }
            }
            feedback.resize(talker_embd, 0.0);

            let talker_pos = Self::qwen3_position(cur_pos as i32, 1);
            talker_batch.clear();
            talker_batch.set_embd(&feedback, &talker_pos, 0);

            self.talker_ctx
                .decode(&mut talker_batch)
                .map_err(|e| format!("Talker step failed: {}", e))?;

            cur_pos += 1;
        }

        // 发送剩余的帧
        let total_frames = all_codes.len() / 16;

        if streaming_decode_enabled {
            if total_frames > last_sent_frame {
                let start_frame = last_sent_frame;
                let end_frame = total_frames;

                let start_idx = start_frame * 16;
                let end_idx = end_frame * 16;
                let frame_codes: Vec<i64> = all_codes[start_idx..end_idx]
                    .iter()
                    .map(|&c| c as i64)
                    .collect();

                let _ = tx.send((frame_codes, true));
            } else {
                let _ = tx.send((Vec::new(), true));
            }
        } else {
            // Keep non-streaming decode off the critical path so talker/predictor
            // generation does not contend with the audio decoder on the same GPU.
            if total_frames == 0 {
                let _ = tx.send((Vec::new(), true));
            } else {
                let total_chunks = total_frames.div_ceil(non_stream_decode_interval);
                match decode_gpu_monitor.as_ref() {
                    Some(monitor) => println!(
                        "Post-EOS decode scheduled: total_frames={} chunk_frames={} chunks={} gpu_device={}",
                        total_frames, non_stream_decode_interval, total_chunks, monitor.label
                    ),
                    None => println!(
                        "Post-EOS decode scheduled: total_frames={} chunk_frames={} chunks={} gpu_telemetry=unavailable",
                        total_frames, non_stream_decode_interval, total_chunks
                    ),
                }

                let mut start_frame = 0;
                while start_frame < total_frames {
                    let end_frame = (start_frame + non_stream_decode_interval).min(total_frames);
                    let is_final = end_frame == total_frames;
                    let start_idx = start_frame * 16;
                    let end_idx = end_frame * 16;
                    let frame_codes: Vec<i64> = all_codes[start_idx..end_idx]
                        .iter()
                        .map(|&c| c as i64)
                        .collect();

                    tx.send((frame_codes, is_final))
                        .map_err(|e| format!("Audio decode scheduling failed: {}", e))?;
                    start_frame = end_frame;
                }
            }
        }

        drop(tx);
        let generation_elapsed = generation_started.elapsed();

        let decoder_result = decoder_handle
            .join()
            .map_err(|_| "Decoder thread panicked".to_string())?;

        let total_elapsed = generation_elapsed + decoder_result.active_decode_time;
        println!(
            "Timing: generation={:.2}s decode={:.2}s total={:.2}s decode_chunks={} decode_frames={} mode={}{}",
            generation_elapsed.as_secs_f64(),
            decoder_result.active_decode_time.as_secs_f64(),
            total_elapsed.as_secs_f64(),
            decoder_result.chunk_count,
            decoder_result.frame_count,
            if streaming_decode_enabled {
                "streaming"
            } else {
                "non-streaming"
            },
            decoder_result.progress_stats.summary_suffix()
        );

        Ok(AudioSample {
            samples: decoder_result.audio,
            sample_rate: 24000,
            channels: 1,
        })
    }
}
