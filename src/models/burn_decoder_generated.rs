// Generated from the normalized Qwen3-TTS decoder ONNX graph by burn-onnx.
use burn::prelude::*;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::nn::conv::ConvTranspose1d;
use burn::nn::conv::ConvTranspose1dConfig;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    constant1: burn::module::Param<Tensor<B, 1>>,
    constant2: burn::module::Param<Tensor<B, 1>>,
    constant3: burn::module::Param<Tensor<B, 1>>,
    constant4: burn::module::Param<Tensor<B, 1>>,
    constant5: burn::module::Param<Tensor<B, 1>>,
    constant6: burn::module::Param<Tensor<B, 1>>,
    constant7: burn::module::Param<Tensor<B, 1>>,
    constant8: burn::module::Param<Tensor<B, 1>>,
    constant9: burn::module::Param<Tensor<B, 1>>,
    constant10: burn::module::Param<Tensor<B, 1>>,
    constant11: burn::module::Param<Tensor<B, 1>>,
    constant12: burn::module::Param<Tensor<B, 1>>,
    constant13: burn::module::Param<Tensor<B, 1>>,
    constant14: burn::module::Param<Tensor<B, 1>>,
    constant15: burn::module::Param<Tensor<B, 1>>,
    constant16: burn::module::Param<Tensor<B, 1>>,
    constant17: burn::module::Param<Tensor<B, 1>>,
    constant18: burn::module::Param<Tensor<B, 1>>,
    constant19: burn::module::Param<Tensor<B, 1>>,
    constant20: burn::module::Param<Tensor<B, 1>>,
    constant21: burn::module::Param<Tensor<B, 1>>,
    constant22: burn::module::Param<Tensor<B, 1>>,
    constant23: burn::module::Param<Tensor<B, 1>>,
    constant24: burn::module::Param<Tensor<B, 1>>,
    constant25: burn::module::Param<Tensor<B, 1>>,
    constant26: burn::module::Param<Tensor<B, 1>>,
    constant27: burn::module::Param<Tensor<B, 1>>,
    constant28: burn::module::Param<Tensor<B, 1>>,
    constant29: burn::module::Param<Tensor<B, 1>>,
    constant30: burn::module::Param<Tensor<B, 1>>,
    constant31: burn::module::Param<Tensor<B, 1>>,
    constant32: burn::module::Param<Tensor<B, 1>>,
    constant33: burn::module::Param<Tensor<B, 1>>,
    constant37: burn::module::Param<Tensor<B, 1>>,
    constant38: burn::module::Param<Tensor<B, 2>>,
    constant40: burn::module::Param<Tensor<B, 1>>,
    constant41: burn::module::Param<Tensor<B, 2>>,
    constant42: burn::module::Param<Tensor<B, 1>>,
    constant43: burn::module::Param<Tensor<B, 2>>,
    constant44: burn::module::Param<Tensor<B, 1>>,
    constant45: burn::module::Param<Tensor<B, 2>>,
    constant46: burn::module::Param<Tensor<B, 1>>,
    constant47: burn::module::Param<Tensor<B, 2>>,
    constant48: burn::module::Param<Tensor<B, 1>>,
    constant49: burn::module::Param<Tensor<B, 2>>,
    constant50: burn::module::Param<Tensor<B, 1>>,
    constant51: burn::module::Param<Tensor<B, 2>>,
    constant52: burn::module::Param<Tensor<B, 1>>,
    constant53: burn::module::Param<Tensor<B, 2>>,
    constant54: burn::module::Param<Tensor<B, 1>>,
    constant55: burn::module::Param<Tensor<B, 2>>,
    constant56: burn::module::Param<Tensor<B, 1>>,
    constant57: burn::module::Param<Tensor<B, 2>>,
    constant58: burn::module::Param<Tensor<B, 1>>,
    constant59: burn::module::Param<Tensor<B, 2>>,
    constant60: burn::module::Param<Tensor<B, 1>>,
    constant61: burn::module::Param<Tensor<B, 2>>,
    constant62: burn::module::Param<Tensor<B, 1>>,
    constant63: burn::module::Param<Tensor<B, 2>>,
    constant64: burn::module::Param<Tensor<B, 1>>,
    constant65: burn::module::Param<Tensor<B, 2>>,
    constant66: burn::module::Param<Tensor<B, 1>>,
    constant67: burn::module::Param<Tensor<B, 2>>,
    constant68: burn::module::Param<Tensor<B, 1>>,
    constant69: burn::module::Param<Tensor<B, 2>>,
    constant74: burn::module::Param<Tensor<B, 1>>,
    constant77: burn::module::Param<Tensor<B, 1>>,
    constant78: burn::module::Param<Tensor<B, 1>>,
    constant83: burn::module::Param<Tensor<B, 1>>,
    constant86: burn::module::Param<Tensor<B, 1>>,
    constant87: burn::module::Param<Tensor<B, 1>>,
    constant212: burn::module::Param<Tensor<B, 3>>,
    constant213: burn::module::Param<Tensor<B, 3>>,
    constant214: burn::module::Param<Tensor<B, 3>>,
    constant215: burn::module::Param<Tensor<B, 3>>,
    constant216: burn::module::Param<Tensor<B, 3>>,
    constant217: burn::module::Param<Tensor<B, 3>>,
    constant218: burn::module::Param<Tensor<B, 3>>,
    constant219: burn::module::Param<Tensor<B, 3>>,
    constant220: burn::module::Param<Tensor<B, 3>>,
    constant221: burn::module::Param<Tensor<B, 3>>,
    constant222: burn::module::Param<Tensor<B, 3>>,
    constant223: burn::module::Param<Tensor<B, 3>>,
    constant224: burn::module::Param<Tensor<B, 3>>,
    constant225: burn::module::Param<Tensor<B, 3>>,
    constant226: burn::module::Param<Tensor<B, 3>>,
    constant227: burn::module::Param<Tensor<B, 3>>,
    constant228: burn::module::Param<Tensor<B, 3>>,
    constant229: burn::module::Param<Tensor<B, 3>>,
    constant230: burn::module::Param<Tensor<B, 3>>,
    constant231: burn::module::Param<Tensor<B, 3>>,
    constant232: burn::module::Param<Tensor<B, 3>>,
    constant233: burn::module::Param<Tensor<B, 3>>,
    constant234: burn::module::Param<Tensor<B, 3>>,
    constant235: burn::module::Param<Tensor<B, 3>>,
    constant236: burn::module::Param<Tensor<B, 3>>,
    constant237: burn::module::Param<Tensor<B, 3>>,
    constant238: burn::module::Param<Tensor<B, 3>>,
    constant239: burn::module::Param<Tensor<B, 3>>,
    constant240: burn::module::Param<Tensor<B, 3>>,
    constant241: burn::module::Param<Tensor<B, 3>>,
    constant242: burn::module::Param<Tensor<B, 3>>,
    constant243: burn::module::Param<Tensor<B, 3>>,
    constant244: burn::module::Param<Tensor<B, 3>>,
    constant245: burn::module::Param<Tensor<B, 3>>,
    constant246: burn::module::Param<Tensor<B, 3>>,
    constant247: burn::module::Param<Tensor<B, 3>>,
    constant248: burn::module::Param<Tensor<B, 3>>,
    constant249: burn::module::Param<Tensor<B, 3>>,
    constant250: burn::module::Param<Tensor<B, 3>>,
    constant251: burn::module::Param<Tensor<B, 3>>,
    constant252: burn::module::Param<Tensor<B, 3>>,
    constant253: burn::module::Param<Tensor<B, 3>>,
    constant254: burn::module::Param<Tensor<B, 3>>,
    constant255: burn::module::Param<Tensor<B, 3>>,
    constant256: burn::module::Param<Tensor<B, 3>>,
    constant257: burn::module::Param<Tensor<B, 3>>,
    constant258: burn::module::Param<Tensor<B, 3>>,
    constant259: burn::module::Param<Tensor<B, 3>>,
    constant260: burn::module::Param<Tensor<B, 3>>,
    constant261: burn::module::Param<Tensor<B, 3>>,
    constant262: burn::module::Param<Tensor<B, 3>>,
    constant263: burn::module::Param<Tensor<B, 3>>,
    constant264: burn::module::Param<Tensor<B, 3>>,
    constant265: burn::module::Param<Tensor<B, 3>>,
    constant266: burn::module::Param<Tensor<B, 3>>,
    constant267: burn::module::Param<Tensor<B, 3>>,
    constant268: burn::module::Param<Tensor<B, 3>>,
    constant269: burn::module::Param<Tensor<B, 3>>,
    conv1d1: Conv1d<B>,
    conv1d2: Conv1d<B>,
    conv1d3: Conv1d<B>,
    linear1: Linear<B>,
    constant428: burn::module::Param<Tensor<B, 3>>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    linear5: Linear<B>,
    linear6: Linear<B>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    linear9: Linear<B>,
    linear10: Linear<B>,
    linear11: Linear<B>,
    linear12: Linear<B>,
    linear13: Linear<B>,
    linear14: Linear<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    linear17: Linear<B>,
    linear18: Linear<B>,
    linear19: Linear<B>,
    linear20: Linear<B>,
    linear21: Linear<B>,
    linear22: Linear<B>,
    linear23: Linear<B>,
    linear24: Linear<B>,
    linear25: Linear<B>,
    linear26: Linear<B>,
    linear27: Linear<B>,
    linear28: Linear<B>,
    linear29: Linear<B>,
    linear30: Linear<B>,
    linear31: Linear<B>,
    linear32: Linear<B>,
    linear33: Linear<B>,
    linear34: Linear<B>,
    linear35: Linear<B>,
    linear36: Linear<B>,
    linear37: Linear<B>,
    linear38: Linear<B>,
    linear39: Linear<B>,
    linear40: Linear<B>,
    linear41: Linear<B>,
    linear42: Linear<B>,
    linear43: Linear<B>,
    linear44: Linear<B>,
    linear45: Linear<B>,
    linear46: Linear<B>,
    linear47: Linear<B>,
    linear48: Linear<B>,
    linear49: Linear<B>,
    linear50: Linear<B>,
    linear51: Linear<B>,
    linear52: Linear<B>,
    linear53: Linear<B>,
    linear54: Linear<B>,
    linear55: Linear<B>,
    linear56: Linear<B>,
    linear57: Linear<B>,
    linear58: Linear<B>,
    constant922: burn::module::Param<Tensor<B, 1, Int>>,
    convtranspose1d1: ConvTranspose1d<B>,
    conv1d4: Conv1d<B>,
    linear59: Linear<B>,
    linear60: Linear<B>,
    convtranspose1d2: ConvTranspose1d<B>,
    conv1d5: Conv1d<B>,
    linear61: Linear<B>,
    linear62: Linear<B>,
    conv1d6: Conv1d<B>,
    convtranspose1d3: ConvTranspose1d<B>,
    conv1d7: Conv1d<B>,
    conv1d8: Conv1d<B>,
    conv1d9: Conv1d<B>,
    conv1d10: Conv1d<B>,
    conv1d11: Conv1d<B>,
    conv1d12: Conv1d<B>,
    convtranspose1d4: ConvTranspose1d<B>,
    conv1d13: Conv1d<B>,
    conv1d14: Conv1d<B>,
    conv1d15: Conv1d<B>,
    conv1d16: Conv1d<B>,
    conv1d17: Conv1d<B>,
    conv1d18: Conv1d<B>,
    convtranspose1d5: ConvTranspose1d<B>,
    conv1d19: Conv1d<B>,
    conv1d20: Conv1d<B>,
    conv1d21: Conv1d<B>,
    conv1d22: Conv1d<B>,
    conv1d23: Conv1d<B>,
    conv1d24: Conv1d<B>,
    convtranspose1d6: ConvTranspose1d<B>,
    conv1d25: Conv1d<B>,
    conv1d26: Conv1d<B>,
    conv1d27: Conv1d<B>,
    conv1d28: Conv1d<B>,
    conv1d29: Conv1d<B>,
    conv1d30: Conv1d<B>,
    conv1d31: Conv1d<B>,
    constant1697: burn::module::Param<Tensor<B, 1, Int>>,
    constant1703: burn::module::Param<Tensor<B, 1, Int>>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        panic!("Use Model::from_file with the generated decoder burnpack path")
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model.load_from(&mut store).expect("Failed to load burnpack file");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant1: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant2: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant3: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant4: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant5: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant6: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant7: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant8: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant9: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant10: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant11: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant12: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant13: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant14: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant15: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant16: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant17: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant18: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant19: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant20: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant21: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant22: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant23: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant24: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant25: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant26: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant27: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant28: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant29: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant30: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant31: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant32: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant33: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([512], device),
            device.clone(),
            false,
            [512].into(),
        );
        let constant37: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant38: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant40: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant41: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant42: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant43: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant44: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant45: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant46: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant47: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant48: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant49: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant50: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant51: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant52: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant53: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant54: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant55: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant56: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant57: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant58: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant59: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant60: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant61: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant62: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant63: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant64: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant65: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant66: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant67: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant68: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([2048], device),
            device.clone(),
            false,
            [2048].into(),
        );
        let constant69: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 2>::zeros([2048, 256], device),
            device.clone(),
            false,
            [2048, 256].into(),
        );
        let constant74: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([1024], device),
            device.clone(),
            false,
            [1024].into(),
        );
        let constant77: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([1024], device),
            device.clone(),
            false,
            [1024].into(),
        );
        let constant78: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([1024], device),
            device.clone(),
            false,
            [1024].into(),
        );
        let constant83: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([1024], device),
            device.clone(),
            false,
            [1024].into(),
        );
        let constant86: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([1024], device),
            device.clone(),
            false,
            [1024].into(),
        );
        let constant87: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([1024], device),
            device.clone(),
            false,
            [1024].into(),
        );
        let constant212: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 1536, 1], device),
            device.clone(),
            false,
            [1, 1536, 1].into(),
        );
        let constant213: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 1536, 1], device),
            device.clone(),
            false,
            [1, 1536, 1].into(),
        );
        let constant214: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant215: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant216: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant217: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant218: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant219: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant220: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant221: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant222: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant223: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant224: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant225: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant226: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant227: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 768, 1], device),
            device.clone(),
            false,
            [1, 768, 1].into(),
        );
        let constant228: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant229: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant230: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant231: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant232: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant233: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant234: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant235: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant236: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant237: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant238: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant239: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant240: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant241: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 384, 1], device),
            device.clone(),
            false,
            [1, 384, 1].into(),
        );
        let constant242: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant243: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant244: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant245: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant246: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant247: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant248: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant249: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant250: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant251: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant252: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant253: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant254: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant255: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 192, 1], device),
            device.clone(),
            false,
            [1, 192, 1].into(),
        );
        let constant256: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant257: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant258: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant259: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant260: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant261: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant262: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant263: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant264: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant265: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant266: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant267: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant268: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let constant269: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 96, 1], device),
            device.clone(),
            false,
            [1, 96, 1].into(),
        );
        let conv1d1 = Conv1dConfig::new(256, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d2 = Conv1dConfig::new(256, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d3 = Conv1dConfig::new(512, 1024, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear1 = LinearConfig::new(1024, 512).with_bias(true).init(device);
        let constant428: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 3>::zeros([1, 32, 1], device),
            device.clone(),
            false,
            [1, 32, 1].into(),
        );
        let linear2 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear3 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear4 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear5 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear6 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear7 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear8 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear9 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear10 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear11 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear12 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear13 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear14 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear15 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear16 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear17 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear18 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear19 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear20 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear21 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear22 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear23 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear24 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear25 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear26 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear27 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear28 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear29 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear30 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear31 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear32 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear33 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear34 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear35 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear36 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear37 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear38 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear39 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear40 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear41 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear42 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear43 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear44 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear45 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear46 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear47 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear48 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear49 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear50 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear51 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear52 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear53 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear54 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear55 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear56 = LinearConfig::new(512, 1024).with_bias(false).init(device);
        let linear57 = LinearConfig::new(1024, 512).with_bias(false).init(device);
        let linear58 = LinearConfig::new(512, 1024).with_bias(true).init(device);
        let constant922: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::zeros([1], device),
            device.clone(),
            false,
            [1].into(),
        );
        let convtranspose1d1 = ConvTranspose1dConfig::new([1024, 1024], 2)
            .with_stride(2)
            .with_padding(0)
            .with_padding_out(0)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d4 = Conv1dConfig::new(1024, 1024, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1024)
            .with_bias(true)
            .init(device);
        let linear59 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear60 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let convtranspose1d2 = ConvTranspose1dConfig::new([1024, 1024], 2)
            .with_stride(2)
            .with_padding(0)
            .with_padding_out(0)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d5 = Conv1dConfig::new(1024, 1024, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1024)
            .with_bias(true)
            .init(device);
        let linear61 = LinearConfig::new(1024, 4096).with_bias(true).init(device);
        let linear62 = LinearConfig::new(4096, 1024).with_bias(true).init(device);
        let conv1d6 = Conv1dConfig::new(1024, 1536, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose1d3 = ConvTranspose1dConfig::new([1536, 768], 16)
            .with_stride(8)
            .with_padding(0)
            .with_padding_out(0)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d7 = Conv1dConfig::new(768, 768, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d8 = Conv1dConfig::new(768, 768, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d9 = Conv1dConfig::new(768, 768, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d10 = Conv1dConfig::new(768, 768, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d11 = Conv1dConfig::new(768, 768, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(9)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d12 = Conv1dConfig::new(768, 768, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose1d4 = ConvTranspose1dConfig::new([768, 384], 10)
            .with_stride(5)
            .with_padding(0)
            .with_padding_out(0)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d13 = Conv1dConfig::new(384, 384, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d14 = Conv1dConfig::new(384, 384, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d15 = Conv1dConfig::new(384, 384, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d16 = Conv1dConfig::new(384, 384, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d17 = Conv1dConfig::new(384, 384, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(9)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d18 = Conv1dConfig::new(384, 384, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose1d5 = ConvTranspose1dConfig::new([384, 192], 8)
            .with_stride(4)
            .with_padding(0)
            .with_padding_out(0)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d19 = Conv1dConfig::new(192, 192, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d20 = Conv1dConfig::new(192, 192, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d21 = Conv1dConfig::new(192, 192, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d22 = Conv1dConfig::new(192, 192, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d23 = Conv1dConfig::new(192, 192, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(9)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d24 = Conv1dConfig::new(192, 192, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose1d6 = ConvTranspose1dConfig::new([192, 96], 6)
            .with_stride(3)
            .with_padding(0)
            .with_padding_out(0)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d25 = Conv1dConfig::new(96, 96, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d26 = Conv1dConfig::new(96, 96, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d27 = Conv1dConfig::new(96, 96, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d28 = Conv1dConfig::new(96, 96, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d29 = Conv1dConfig::new(96, 96, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(9)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d30 = Conv1dConfig::new(96, 96, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d31 = Conv1dConfig::new(96, 1, 7)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant1697: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::zeros([1], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant1703: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::zeros([1], device),
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            constant1,
            constant2,
            constant3,
            constant4,
            constant5,
            constant6,
            constant7,
            constant8,
            constant9,
            constant10,
            constant11,
            constant12,
            constant13,
            constant14,
            constant15,
            constant16,
            constant17,
            constant18,
            constant19,
            constant20,
            constant21,
            constant22,
            constant23,
            constant24,
            constant25,
            constant26,
            constant27,
            constant28,
            constant29,
            constant30,
            constant31,
            constant32,
            constant33,
            constant37,
            constant38,
            constant40,
            constant41,
            constant42,
            constant43,
            constant44,
            constant45,
            constant46,
            constant47,
            constant48,
            constant49,
            constant50,
            constant51,
            constant52,
            constant53,
            constant54,
            constant55,
            constant56,
            constant57,
            constant58,
            constant59,
            constant60,
            constant61,
            constant62,
            constant63,
            constant64,
            constant65,
            constant66,
            constant67,
            constant68,
            constant69,
            constant74,
            constant77,
            constant78,
            constant83,
            constant86,
            constant87,
            constant212,
            constant213,
            constant214,
            constant215,
            constant216,
            constant217,
            constant218,
            constant219,
            constant220,
            constant221,
            constant222,
            constant223,
            constant224,
            constant225,
            constant226,
            constant227,
            constant228,
            constant229,
            constant230,
            constant231,
            constant232,
            constant233,
            constant234,
            constant235,
            constant236,
            constant237,
            constant238,
            constant239,
            constant240,
            constant241,
            constant242,
            constant243,
            constant244,
            constant245,
            constant246,
            constant247,
            constant248,
            constant249,
            constant250,
            constant251,
            constant252,
            constant253,
            constant254,
            constant255,
            constant256,
            constant257,
            constant258,
            constant259,
            constant260,
            constant261,
            constant262,
            constant263,
            constant264,
            constant265,
            constant266,
            constant267,
            constant268,
            constant269,
            conv1d1,
            conv1d2,
            conv1d3,
            linear1,
            constant428,
            linear2,
            linear3,
            linear4,
            linear5,
            linear6,
            linear7,
            linear8,
            linear9,
            linear10,
            linear11,
            linear12,
            linear13,
            linear14,
            linear15,
            linear16,
            linear17,
            linear18,
            linear19,
            linear20,
            linear21,
            linear22,
            linear23,
            linear24,
            linear25,
            linear26,
            linear27,
            linear28,
            linear29,
            linear30,
            linear31,
            linear32,
            linear33,
            linear34,
            linear35,
            linear36,
            linear37,
            linear38,
            linear39,
            linear40,
            linear41,
            linear42,
            linear43,
            linear44,
            linear45,
            linear46,
            linear47,
            linear48,
            linear49,
            linear50,
            linear51,
            linear52,
            linear53,
            linear54,
            linear55,
            linear56,
            linear57,
            linear58,
            constant922,
            convtranspose1d1,
            conv1d4,
            linear59,
            linear60,
            convtranspose1d2,
            conv1d5,
            linear61,
            linear62,
            conv1d6,
            convtranspose1d3,
            conv1d7,
            conv1d8,
            conv1d9,
            conv1d10,
            conv1d11,
            conv1d12,
            convtranspose1d4,
            conv1d13,
            conv1d14,
            conv1d15,
            conv1d16,
            conv1d17,
            conv1d18,
            convtranspose1d5,
            conv1d19,
            conv1d20,
            conv1d21,
            conv1d22,
            conv1d23,
            conv1d24,
            convtranspose1d6,
            conv1d25,
            conv1d26,
            conv1d27,
            conv1d28,
            conv1d29,
            conv1d30,
            conv1d31,
            constant1697,
            constant1703,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        audio_codes: Tensor<B, 3, Int>,
        is_last: Tensor<B, 1>,
        pre_conv_history: Tensor<B, 3>,
        latent_buffer: Tensor<B, 3>,
        conv_history: Tensor<B, 3>,
        past_key_0: Tensor<B, 4>,
        past_key_1: Tensor<B, 4>,
        past_key_2: Tensor<B, 4>,
        past_key_3: Tensor<B, 4>,
        past_key_4: Tensor<B, 4>,
        past_key_5: Tensor<B, 4>,
        past_key_6: Tensor<B, 4>,
        past_key_7: Tensor<B, 4>,
        past_value_0: Tensor<B, 4>,
        past_value_1: Tensor<B, 4>,
        past_value_2: Tensor<B, 4>,
        past_value_3: Tensor<B, 4>,
        past_value_4: Tensor<B, 4>,
        past_value_5: Tensor<B, 4>,
        past_value_6: Tensor<B, 4>,
        past_value_7: Tensor<B, 4>,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 1, Int>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
    ) {
        let constant1_out1 = self.constant1.val();
        let constant2_out1 = self.constant2.val();
        let constant3_out1 = self.constant3.val();
        let constant4_out1 = self.constant4.val();
        let constant5_out1 = self.constant5.val();
        let constant6_out1 = self.constant6.val();
        let constant7_out1 = self.constant7.val();
        let constant8_out1 = self.constant8.val();
        let constant9_out1 = self.constant9.val();
        let constant10_out1 = self.constant10.val();
        let constant11_out1 = self.constant11.val();
        let constant12_out1 = self.constant12.val();
        let constant13_out1 = self.constant13.val();
        let constant14_out1 = self.constant14.val();
        let constant15_out1 = self.constant15.val();
        let constant16_out1 = self.constant16.val();
        let constant17_out1 = self.constant17.val();
        let constant18_out1 = self.constant18.val();
        let constant19_out1 = self.constant19.val();
        let constant20_out1 = self.constant20.val();
        let constant21_out1 = self.constant21.val();
        let constant22_out1 = self.constant22.val();
        let constant23_out1 = self.constant23.val();
        let constant24_out1 = self.constant24.val();
        let constant25_out1 = self.constant25.val();
        let constant26_out1 = self.constant26.val();
        let constant27_out1 = self.constant27.val();
        let constant28_out1 = self.constant28.val();
        let constant29_out1 = self.constant29.val();
        let constant30_out1 = self.constant30.val();
        let constant31_out1 = self.constant31.val();
        let constant32_out1 = self.constant32.val();
        let constant33_out1 = self.constant33.val();
        let constant37_out1 = self.constant37.val();
        let constant38_out1 = self.constant38.val();
        let constant40_out1 = self.constant40.val();
        let constant41_out1 = self.constant41.val();
        let constant42_out1 = self.constant42.val();
        let constant43_out1 = self.constant43.val();
        let constant44_out1 = self.constant44.val();
        let constant45_out1 = self.constant45.val();
        let constant46_out1 = self.constant46.val();
        let constant47_out1 = self.constant47.val();
        let constant48_out1 = self.constant48.val();
        let constant49_out1 = self.constant49.val();
        let constant50_out1 = self.constant50.val();
        let constant51_out1 = self.constant51.val();
        let constant52_out1 = self.constant52.val();
        let constant53_out1 = self.constant53.val();
        let constant54_out1 = self.constant54.val();
        let constant55_out1 = self.constant55.val();
        let constant56_out1 = self.constant56.val();
        let constant57_out1 = self.constant57.val();
        let constant58_out1 = self.constant58.val();
        let constant59_out1 = self.constant59.val();
        let constant60_out1 = self.constant60.val();
        let constant61_out1 = self.constant61.val();
        let constant62_out1 = self.constant62.val();
        let constant63_out1 = self.constant63.val();
        let constant64_out1 = self.constant64.val();
        let constant65_out1 = self.constant65.val();
        let constant66_out1 = self.constant66.val();
        let constant67_out1 = self.constant67.val();
        let constant68_out1 = self.constant68.val();
        let constant69_out1 = self.constant69.val();
        let constant74_out1 = self.constant74.val();
        let constant77_out1 = self.constant77.val();
        let constant78_out1 = self.constant78.val();
        let constant83_out1 = self.constant83.val();
        let constant86_out1 = self.constant86.val();
        let constant87_out1 = self.constant87.val();
        let constant212_out1 = self.constant212.val();
        let constant213_out1 = self.constant213.val();
        let constant214_out1 = self.constant214.val();
        let constant215_out1 = self.constant215.val();
        let constant216_out1 = self.constant216.val();
        let constant217_out1 = self.constant217.val();
        let constant218_out1 = self.constant218.val();
        let constant219_out1 = self.constant219.val();
        let constant220_out1 = self.constant220.val();
        let constant221_out1 = self.constant221.val();
        let constant222_out1 = self.constant222.val();
        let constant223_out1 = self.constant223.val();
        let constant224_out1 = self.constant224.val();
        let constant225_out1 = self.constant225.val();
        let constant226_out1 = self.constant226.val();
        let constant227_out1 = self.constant227.val();
        let constant228_out1 = self.constant228.val();
        let constant229_out1 = self.constant229.val();
        let constant230_out1 = self.constant230.val();
        let constant231_out1 = self.constant231.val();
        let constant232_out1 = self.constant232.val();
        let constant233_out1 = self.constant233.val();
        let constant234_out1 = self.constant234.val();
        let constant235_out1 = self.constant235.val();
        let constant236_out1 = self.constant236.val();
        let constant237_out1 = self.constant237.val();
        let constant238_out1 = self.constant238.val();
        let constant239_out1 = self.constant239.val();
        let constant240_out1 = self.constant240.val();
        let constant241_out1 = self.constant241.val();
        let constant242_out1 = self.constant242.val();
        let constant243_out1 = self.constant243.val();
        let constant244_out1 = self.constant244.val();
        let constant245_out1 = self.constant245.val();
        let constant246_out1 = self.constant246.val();
        let constant247_out1 = self.constant247.val();
        let constant248_out1 = self.constant248.val();
        let constant249_out1 = self.constant249.val();
        let constant250_out1 = self.constant250.val();
        let constant251_out1 = self.constant251.val();
        let constant252_out1 = self.constant252.val();
        let constant253_out1 = self.constant253.val();
        let constant254_out1 = self.constant254.val();
        let constant255_out1 = self.constant255.val();
        let constant256_out1 = self.constant256.val();
        let constant257_out1 = self.constant257.val();
        let constant258_out1 = self.constant258.val();
        let constant259_out1 = self.constant259.val();
        let constant260_out1 = self.constant260.val();
        let constant261_out1 = self.constant261.val();
        let constant262_out1 = self.constant262.val();
        let constant263_out1 = self.constant263.val();
        let constant264_out1 = self.constant264.val();
        let constant265_out1 = self.constant265.val();
        let constant266_out1 = self.constant266.val();
        let constant267_out1 = self.constant267.val();
        let constant268_out1 = self.constant268.val();
        let constant269_out1 = self.constant269.val();
        let shape1_out1: [i64; 3] = {
            let axes = &audio_codes.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant270_out1 = 1i64;
        let actual_idx = if constant270_out1 < 0 {
            (shape1_out1.len() as i64 + constant270_out1) as usize
        } else {
            constant270_out1 as usize
        };
        let gather1_out1 = shape1_out1[actual_idx] as i64;
        let transpose1_out1 = audio_codes.permute([0, 2, 1]);
        let constant271_out1 = 0i64;
        let slice1_out1 = transpose1_out1.clone().slice(s![.., 0..1, ..]);
        let transpose2_out1 = slice1_out1.permute([1, 0, 2]);
        let squeeze1_out1 = transpose2_out1.squeeze_dims::<2>(&[0]);
        let clip1_out1 = constant37_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze1_out1: Tensor<B, 2> = clip1_out1.unsqueeze_dims::<2>(&[1]);
        let div1_out1 = constant38_out1.div(unsqueeze1_out1);
        let gather2_out1 = div1_out1.take::<2, 3>(0, squeeze1_out1);
        let transpose3_out1 = gather2_out1.permute([0, 2, 1]);
        let conv1d1_out1 = self.conv1d1.forward(transpose3_out1);
        let slice2_out1 = transpose1_out1.slice(s![.., 1.., ..]);
        let transpose4_out1 = slice2_out1.permute([1, 0, 2]);
        let slice3_out1 = transpose4_out1.clone().slice(s![0..1, .., ..]);
        let slice4_out1 = transpose4_out1.clone().slice(s![1..2, .., ..]);
        let slice5_out1 = transpose4_out1.clone().slice(s![2..3, .., ..]);
        let slice6_out1 = transpose4_out1.clone().slice(s![3..4, .., ..]);
        let slice7_out1 = transpose4_out1.clone().slice(s![4..5, .., ..]);
        let slice8_out1 = transpose4_out1.clone().slice(s![5..6, .., ..]);
        let slice9_out1 = transpose4_out1.clone().slice(s![6..7, .., ..]);
        let slice10_out1 = transpose4_out1.clone().slice(s![7..8, .., ..]);
        let slice11_out1 = transpose4_out1.clone().slice(s![8..9, .., ..]);
        let slice12_out1 = transpose4_out1.clone().slice(s![9..10, .., ..]);
        let slice13_out1 = transpose4_out1.clone().slice(s![10..11, .., ..]);
        let slice14_out1 = transpose4_out1.clone().slice(s![11..12, .., ..]);
        let slice15_out1 = transpose4_out1.clone().slice(s![12..13, .., ..]);
        let slice16_out1 = transpose4_out1.clone().slice(s![13..14, .., ..]);
        let slice17_out1 = transpose4_out1.slice(s![14..15, .., ..]);
        let squeeze2_out1 = slice3_out1.squeeze_dims::<2>(&[0]);
        let squeeze3_out1 = slice4_out1.squeeze_dims::<2>(&[0]);
        let squeeze4_out1 = slice5_out1.squeeze_dims::<2>(&[0]);
        let squeeze5_out1 = slice6_out1.squeeze_dims::<2>(&[0]);
        let squeeze6_out1 = slice7_out1.squeeze_dims::<2>(&[0]);
        let squeeze7_out1 = slice8_out1.squeeze_dims::<2>(&[0]);
        let squeeze8_out1 = slice9_out1.squeeze_dims::<2>(&[0]);
        let squeeze9_out1 = slice10_out1.squeeze_dims::<2>(&[0]);
        let squeeze10_out1 = slice11_out1.squeeze_dims::<2>(&[0]);
        let squeeze11_out1 = slice12_out1.squeeze_dims::<2>(&[0]);
        let squeeze12_out1 = slice13_out1.squeeze_dims::<2>(&[0]);
        let squeeze13_out1 = slice14_out1.squeeze_dims::<2>(&[0]);
        let squeeze14_out1 = slice15_out1.squeeze_dims::<2>(&[0]);
        let squeeze15_out1 = slice16_out1.squeeze_dims::<2>(&[0]);
        let squeeze16_out1 = slice17_out1.squeeze_dims::<2>(&[0]);
        let clip2_out1 = constant40_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze2_out1: Tensor<B, 2> = clip2_out1.unsqueeze_dims::<2>(&[1]);
        let div2_out1 = constant41_out1.div(unsqueeze2_out1);
        let gather3_out1 = div2_out1.take::<2, 3>(0, squeeze2_out1);
        let transpose5_out1 = gather3_out1.permute([0, 2, 1]);
        let clip3_out1 = constant42_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze3_out1: Tensor<B, 2> = clip3_out1.unsqueeze_dims::<2>(&[1]);
        let div3_out1 = constant43_out1.div(unsqueeze3_out1);
        let gather4_out1 = div3_out1.take::<2, 3>(0, squeeze3_out1);
        let transpose6_out1 = gather4_out1.permute([0, 2, 1]);
        let add3_out1 = transpose5_out1.add(transpose6_out1);
        let clip4_out1 = constant44_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze4_out1: Tensor<B, 2> = clip4_out1.unsqueeze_dims::<2>(&[1]);
        let div4_out1 = constant45_out1.div(unsqueeze4_out1);
        let gather5_out1 = div4_out1.take::<2, 3>(0, squeeze4_out1);
        let transpose7_out1 = gather5_out1.permute([0, 2, 1]);
        let add4_out1 = add3_out1.add(transpose7_out1);
        let clip5_out1 = constant46_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze5_out1: Tensor<B, 2> = clip5_out1.unsqueeze_dims::<2>(&[1]);
        let div5_out1 = constant47_out1.div(unsqueeze5_out1);
        let gather6_out1 = div5_out1.take::<2, 3>(0, squeeze5_out1);
        let transpose8_out1 = gather6_out1.permute([0, 2, 1]);
        let add5_out1 = add4_out1.add(transpose8_out1);
        let clip6_out1 = constant48_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze6_out1: Tensor<B, 2> = clip6_out1.unsqueeze_dims::<2>(&[1]);
        let div6_out1 = constant49_out1.div(unsqueeze6_out1);
        let gather7_out1 = div6_out1.take::<2, 3>(0, squeeze6_out1);
        let transpose9_out1 = gather7_out1.permute([0, 2, 1]);
        let add6_out1 = add5_out1.add(transpose9_out1);
        let clip7_out1 = constant50_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze7_out1: Tensor<B, 2> = clip7_out1.unsqueeze_dims::<2>(&[1]);
        let div7_out1 = constant51_out1.div(unsqueeze7_out1);
        let gather8_out1 = div7_out1.take::<2, 3>(0, squeeze7_out1);
        let transpose10_out1 = gather8_out1.permute([0, 2, 1]);
        let add7_out1 = add6_out1.add(transpose10_out1);
        let clip8_out1 = constant52_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze8_out1: Tensor<B, 2> = clip8_out1.unsqueeze_dims::<2>(&[1]);
        let div8_out1 = constant53_out1.div(unsqueeze8_out1);
        let gather9_out1 = div8_out1.take::<2, 3>(0, squeeze8_out1);
        let transpose11_out1 = gather9_out1.permute([0, 2, 1]);
        let add8_out1 = add7_out1.add(transpose11_out1);
        let clip9_out1 = constant54_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze9_out1: Tensor<B, 2> = clip9_out1.unsqueeze_dims::<2>(&[1]);
        let div9_out1 = constant55_out1.div(unsqueeze9_out1);
        let gather10_out1 = div9_out1.take::<2, 3>(0, squeeze9_out1);
        let transpose12_out1 = gather10_out1.permute([0, 2, 1]);
        let add9_out1 = add8_out1.add(transpose12_out1);
        let clip10_out1 = constant56_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze10_out1: Tensor<B, 2> = clip10_out1.unsqueeze_dims::<2>(&[1]);
        let div10_out1 = constant57_out1.div(unsqueeze10_out1);
        let gather11_out1 = div10_out1.take::<2, 3>(0, squeeze10_out1);
        let transpose13_out1 = gather11_out1.permute([0, 2, 1]);
        let add10_out1 = add9_out1.add(transpose13_out1);
        let clip11_out1 = constant58_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze11_out1: Tensor<B, 2> = clip11_out1.unsqueeze_dims::<2>(&[1]);
        let div11_out1 = constant59_out1.div(unsqueeze11_out1);
        let gather12_out1 = div11_out1.take::<2, 3>(0, squeeze11_out1);
        let transpose14_out1 = gather12_out1.permute([0, 2, 1]);
        let add11_out1 = add10_out1.add(transpose14_out1);
        let clip12_out1 = constant60_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze12_out1: Tensor<B, 2> = clip12_out1.unsqueeze_dims::<2>(&[1]);
        let div12_out1 = constant61_out1.div(unsqueeze12_out1);
        let gather13_out1 = div12_out1.take::<2, 3>(0, squeeze12_out1);
        let transpose15_out1 = gather13_out1.permute([0, 2, 1]);
        let add12_out1 = add11_out1.add(transpose15_out1);
        let clip13_out1 = constant62_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze13_out1: Tensor<B, 2> = clip13_out1.unsqueeze_dims::<2>(&[1]);
        let div13_out1 = constant63_out1.div(unsqueeze13_out1);
        let gather14_out1 = div13_out1.take::<2, 3>(0, squeeze13_out1);
        let transpose16_out1 = gather14_out1.permute([0, 2, 1]);
        let add13_out1 = add12_out1.add(transpose16_out1);
        let clip14_out1 = constant64_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze14_out1: Tensor<B, 2> = clip14_out1.unsqueeze_dims::<2>(&[1]);
        let div14_out1 = constant65_out1.div(unsqueeze14_out1);
        let gather15_out1 = div14_out1.take::<2, 3>(0, squeeze14_out1);
        let transpose17_out1 = gather15_out1.permute([0, 2, 1]);
        let add14_out1 = add13_out1.add(transpose17_out1);
        let clip15_out1 = constant66_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze15_out1: Tensor<B, 2> = clip15_out1.unsqueeze_dims::<2>(&[1]);
        let div15_out1 = constant67_out1.div(unsqueeze15_out1);
        let gather16_out1 = div15_out1.take::<2, 3>(0, squeeze15_out1);
        let transpose18_out1 = gather16_out1.permute([0, 2, 1]);
        let add15_out1 = add14_out1.add(transpose18_out1);
        let clip16_out1 = constant68_out1.clamp_min(0.000009999999747378752f64);
        let unsqueeze16_out1: Tensor<B, 2> = clip16_out1.unsqueeze_dims::<2>(&[1]);
        let div16_out1 = constant69_out1.div(unsqueeze16_out1);
        let gather17_out1 = div16_out1.take::<2, 3>(0, squeeze16_out1);
        let transpose19_out1 = gather17_out1.permute([0, 2, 1]);
        let add16_out1 = add15_out1.add(transpose19_out1);
        let conv1d2_out1 = self.conv1d2.forward(add16_out1);
        let add17_out1 = conv1d1_out1.add(conv1d2_out1);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [pre_conv_history.clone(), add17_out1.clone()].into(),
            2,
        );
        let shape2_out1: [i64; 3] = {
            let axes = &pre_conv_history.dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant392_out1 = 2i64;
        let actual_idx = if constant392_out1 < 0 {
            (shape2_out1.len() as i64 + constant392_out1) as usize
        } else {
            constant392_out1 as usize
        };
        let gather18_out1 = shape2_out1[actual_idx] as i64;
        let pad1_out1 = concat1_out1
            .pad((2, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d3_out1 = self.conv1d3.forward(pad1_out1);
        let gather21_out1 = gather18_out1;
        let unsqueeze19_out1 = [gather21_out1];
        let slice19_out1 = conv1d3_out1
            .slice(s![.., .., unsqueeze19_out1[0]..9223372036854775807]);
        let transpose21_out1 = slice19_out1.permute([0, 2, 1]);
        let slice20_out1 = add17_out1.slice(s![.., .., - 2..]);
        let linear1_out1 = self.linear1.forward(transpose21_out1);
        let shape5_out1: [i64; 4] = {
            let axes = &past_key_0.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant425_out1 = 2i64;
        let actual_idx = if constant425_out1 < 0 {
            (shape5_out1.len() as i64 + constant425_out1) as usize
        } else {
            constant425_out1 as usize
        };
        let gather22_out1 = shape5_out1[actual_idx] as i64;
        let add23_out1 = gather22_out1 + gather1_out1;
        let range1_out1 = Tensor::arange_step(
            gather22_out1..add23_out1,
            1i64 as usize,
            &*self.device,
        );
        let unsqueeze20_out1: Tensor<B, 2, Int> = range1_out1.unsqueeze_dims::<2>(&[0]);
        let constant428_out1 = self.constant428.val();
        let shape6_out1: [i64; 2] = {
            let axes = &unsqueeze20_out1.clone().dims()[0..2];
            let mut output = [0i64; 2];
            for i in 0..2 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant429_out1 = 0i64;
        let actual_idx = if constant429_out1 < 0 {
            (shape6_out1.len() as i64 + constant429_out1) as usize
        } else {
            constant429_out1 as usize
        };
        let gather23_out1 = shape6_out1[actual_idx] as i64;
        let unsqueeze21_out1 = [gather23_out1];
        let constant431_out1: [i64; 1] = [-1i64];
        let constant432_out1: [i64; 1] = [1i64];
        let concat5_out1: [i64; 3usize] = [
            &unsqueeze21_out1[..],
            &constant431_out1[..],
            &constant432_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape3_out1 = concat5_out1;
        let shape7_out1: [i64; 1] = [3i64];
        let constantofshape2_out1 = Tensor::<
            B,
            1,
            Int,
        >::from_data_dtype(
                burn::tensor::TensorData::from([1i64 as i64]),
                &*self.device,
                burn::tensor::DType::I64,
            )
            .reshape([1])
            .expand(shape7_out1);
        let constant434_out1 = -1i64;
        let mul1_out1 = constantofshape2_out1.clone().mul_scalar(constant434_out1);
        let equal1_out1 = {
            let shape_tensor = Tensor::<
                B,
                1,
                Int,
            >::from_data_dtype(
                burn::tensor::TensorData::from(reshape3_out1.as_slice()),
                &*self.device,
                burn::tensor::DType::I64,
            );
            shape_tensor.equal(mul1_out1)
        };
        let where1_out1 = Tensor::<
            B,
            1,
            burn::tensor::Int,
        >::from_data_dtype(
                burn::tensor::TensorData::from(&reshape3_out1 as &[i64]),
                &*self.device,
                burn::tensor::DType::I64,
            )
            .mask_where(equal1_out1, constantofshape2_out1);
        let expand1_out1 = {
            let onnx_shape: [i64; 3usize] = TryInto::<
                [i64; 3usize],
            >::try_into(where1_out1.to_data().convert::<i64>().as_slice().unwrap())
                .unwrap();
            let input_dims = constant428_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..3usize {
                let dim_offset = 3usize - 3usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            constant428_out1.expand(shape)
        };
        let unsqueeze22_out1: Tensor<B, 3, Int> = unsqueeze20_out1
            .unsqueeze_dims::<3>(&[1]);
        let cast8_out1 = unsqueeze22_out1.float();
        let matmul2_out1 = expand1_out1.matmul(cast8_out1);
        let transpose22_out1 = matmul2_out1.permute([0, 2, 1]);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [transpose22_out1.clone(), transpose22_out1].into(),
            2,
        );
        let cos1_out1 = concat6_out1.clone().cos();
        let sin1_out1 = concat6_out1.sin();
        let range2_out1 = Tensor::arange_step(
            0i64..gather1_out1,
            1i64 as usize,
            &*self.device,
        );
        let unsqueeze23_out1: Tensor<B, 2, Int> = range2_out1.unsqueeze_dims::<2>(&[1]);
        let range3_out1 = Tensor::arange_step(
            0i64..add23_out1,
            1i64 as usize,
            &*self.device,
        );
        let unsqueeze24_out1: Tensor<B, 2, Int> = range3_out1.unsqueeze_dims::<2>(&[0]);
        let add24_out1 = unsqueeze23_out1.add_scalar(gather22_out1);
        let lessorequal1_out1 = unsqueeze24_out1.clone().lower_equal(add24_out1.clone());
        let constant444_out1 = 72i64;
        let sub6_out1 = add24_out1.sub_scalar(constant444_out1);
        let greater1_out1 = unsqueeze24_out1.greater(sub6_out1);
        let and1_out1 = lessorequal1_out1.bool_and(greater1_out1);
        let constant445_out1 = 0f32;
        let constant446_out1 = f32::NEG_INFINITY;
        let where2_out1 = Tensor::<
            B,
            1,
        >::from_data_dtype(
                burn::tensor::TensorData::from([constant446_out1 as f64]),
                &*self.device,
                burn::tensor::DType::F32,
            )
            .reshape([1, 1])
            .mask_fill(and1_out1, constant445_out1);
        let unsqueeze25_out1: Tensor<B, 3> = where2_out1.unsqueeze_dims::<3>(&[0]);
        let unsqueeze26_out1: Tensor<B, 4> = unsqueeze25_out1.unsqueeze_dims::<4>(&[0]);
        let constant449_out1 = 2f32;
        let pow1_out1 = linear1_out1.clone().powf_scalar(constant449_out1);
        let reducemean1_out1 = { pow1_out1.mean_dim(2usize) };
        let constant450_out1 = 0.00001f32;
        let add25_out1 = reducemean1_out1.add_scalar(constant450_out1);
        let sqrt1_out1 = add25_out1.sqrt();
        let constant451_out1 = 1f32;
        let div18_out1 = constant451_out1 / sqrt1_out1;
        let mul4_out1 = linear1_out1.clone().mul(div18_out1);
        let mul5_out1 = constant1_out1.unsqueeze_dims(&[0isize, 1isize]).mul(mul4_out1);
        let shape8_out1: [i64; 3] = {
            let axes = &mul5_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant452_out1 = 0i64;
        let actual_idx = if constant452_out1 < 0 {
            (shape8_out1.len() as i64 + constant452_out1) as usize
        } else {
            constant452_out1 as usize
        };
        let gather24_out1 = shape8_out1[actual_idx] as i64;
        let constant453_out1 = 1i64;
        let actual_idx = if constant453_out1 < 0 {
            (shape8_out1.len() as i64 + constant453_out1) as usize
        } else {
            constant453_out1 as usize
        };
        let gather25_out1 = shape8_out1[actual_idx] as i64;
        let linear2_out1 = self.linear2.forward(mul5_out1.clone());
        let unsqueeze27_out1 = [gather24_out1];
        let unsqueeze28_out1 = [gather25_out1];
        let constant456_out1: [i64; 1] = [-1i64];
        let constant457_out1: [i64; 1] = [64i64];
        let concat7_out1: [i64; 4usize] = [
            &unsqueeze27_out1[..],
            &unsqueeze28_out1[..],
            &constant456_out1[..],
            &constant457_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze29_out1 = [gather24_out1];
        let unsqueeze30_out1 = [gather25_out1];
        let constant460_out1: [i64; 1] = [-1i64];
        let constant461_out1: [i64; 1] = [64i64];
        let concat8_out1: [i64; 4usize] = [
            &unsqueeze29_out1[..],
            &unsqueeze30_out1[..],
            &constant460_out1[..],
            &constant461_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze31_out1 = [gather24_out1];
        let unsqueeze32_out1 = [gather25_out1];
        let constant464_out1: [i64; 1] = [-1i64];
        let constant465_out1: [i64; 1] = [64i64];
        let concat9_out1: [i64; 4usize] = [
            &unsqueeze31_out1[..],
            &unsqueeze32_out1[..],
            &constant464_out1[..],
            &constant465_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape4_out1 = linear2_out1.reshape(concat7_out1);
        let transpose23_out1 = reshape4_out1.permute([0, 2, 1, 3]);
        let linear3_out1 = self.linear3.forward(mul5_out1.clone());
        let reshape5_out1 = linear3_out1.reshape(concat8_out1);
        let transpose24_out1 = reshape5_out1.permute([0, 2, 1, 3]);
        let linear4_out1 = self.linear4.forward(mul5_out1);
        let reshape6_out1 = linear4_out1.reshape(concat9_out1);
        let transpose25_out1 = reshape6_out1.permute([0, 2, 1, 3]);
        let unsqueeze33_out1: Tensor<B, 4> = cos1_out1.unsqueeze_dims::<4>(&[1]);
        let unsqueeze34_out1: Tensor<B, 4> = sin1_out1.unsqueeze_dims::<4>(&[1]);
        let mul6_out1 = transpose23_out1.clone().mul(unsqueeze33_out1.clone());
        let shape10_out1: [i64; 4] = {
            let axes = &transpose23_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant468_out1 = 3i64;
        let actual_idx = if constant468_out1 < 0 {
            (shape10_out1.len() as i64 + constant468_out1) as usize
        } else {
            constant468_out1 as usize
        };
        let gather26_out1 = shape10_out1[actual_idx] as i64;
        let constant469_out1 = 2i64;
        let div19_out1 = gather26_out1 / constant469_out1;
        let unsqueeze35_out1 = [div19_out1];
        let slice21_out1 = transpose23_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze35_out1[0]]);
        let unsqueeze36_out1 = [div19_out1];
        let slice22_out1 = transpose23_out1
            .slice(s![.., .., .., unsqueeze36_out1[0]..9223372036854775807]);
        let neg1_out1 = slice22_out1.neg();
        let concat10_out1 = burn::tensor::Tensor::cat(
            [neg1_out1, slice21_out1].into(),
            3,
        );
        let mul7_out1 = concat10_out1.mul(unsqueeze34_out1.clone());
        let add26_out1 = mul6_out1.add(mul7_out1);
        let mul8_out1 = transpose24_out1.clone().mul(unsqueeze33_out1.clone());
        let shape11_out1: [i64; 4] = {
            let axes = &transpose24_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant478_out1 = 3i64;
        let actual_idx = if constant478_out1 < 0 {
            (shape11_out1.len() as i64 + constant478_out1) as usize
        } else {
            constant478_out1 as usize
        };
        let gather27_out1 = shape11_out1[actual_idx] as i64;
        let constant479_out1 = 2i64;
        let div20_out1 = gather27_out1 / constant479_out1;
        let unsqueeze37_out1 = [div20_out1];
        let slice23_out1 = transpose24_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze37_out1[0]]);
        let unsqueeze38_out1 = [div20_out1];
        let slice24_out1 = transpose24_out1
            .slice(s![.., .., .., unsqueeze38_out1[0]..9223372036854775807]);
        let neg2_out1 = slice24_out1.neg();
        let concat11_out1 = burn::tensor::Tensor::cat(
            [neg2_out1, slice23_out1].into(),
            3,
        );
        let mul9_out1 = concat11_out1.mul(unsqueeze34_out1.clone());
        let add27_out1 = mul8_out1.add(mul9_out1);
        let concat12_out1 = burn::tensor::Tensor::cat(
            [past_key_0, add27_out1].into(),
            2,
        );
        let concat13_out1 = burn::tensor::Tensor::cat(
            [past_value_0, transpose25_out1].into(),
            2,
        );
        let slice25_out1 = concat12_out1.slice(s![.., .., - 72.., ..]);
        let slice26_out1 = concat13_out1.slice(s![.., .., - 72.., ..]);
        let shape12_out1: [i64; 4] = {
            let axes = &slice25_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant496_out1 = 2i64;
        let actual_idx = if constant496_out1 < 0 {
            (shape12_out1.len() as i64 + constant496_out1) as usize
        } else {
            constant496_out1 as usize
        };
        let gather28_out1 = shape12_out1[actual_idx] as i64;
        let unsqueeze39_out1 = [gather28_out1];
        let slice27_out1 = unsqueeze26_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze39_out1[0]]);
        let transpose26_out1 = slice25_out1.clone().permute([0, 1, 3, 2]);
        let constant501_out1 = 0.35355338f32;
        let mul10_out1 = add26_out1.mul_scalar(constant501_out1);
        let constant502_out1 = 0.35355338f32;
        let mul11_out1 = transpose26_out1.mul_scalar(constant502_out1);
        let matmul6_out1 = mul10_out1.matmul(mul11_out1);
        let add28_out1 = matmul6_out1.add(slice27_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(add28_out1, 3);
        let matmul7_out1 = softmax1_out1.matmul(slice26_out1.clone());
        let transpose27_out1 = matmul7_out1.permute([0, 2, 1, 3]);
        let unsqueeze40_out1 = [gather24_out1];
        let unsqueeze41_out1 = [gather25_out1];
        let constant505_out1: [i64; 1] = [-1i64];
        let concat14_out1: [i64; 3usize] = [
            &unsqueeze40_out1[..],
            &unsqueeze41_out1[..],
            &constant505_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape7_out1 = transpose27_out1.reshape(concat14_out1);
        let linear5_out1 = self.linear5.forward(reshape7_out1);
        let mul12_out1 = constant3_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear5_out1);
        let add29_out1 = linear1_out1.add(mul12_out1);
        let constant506_out1 = 2f32;
        let pow2_out1 = add29_out1.clone().powf_scalar(constant506_out1);
        let reducemean2_out1 = { pow2_out1.mean_dim(2usize) };
        let constant507_out1 = 0.00001f32;
        let add30_out1 = reducemean2_out1.add_scalar(constant507_out1);
        let sqrt2_out1 = add30_out1.sqrt();
        let constant508_out1 = 1f32;
        let div21_out1 = constant508_out1 / sqrt2_out1;
        let mul13_out1 = add29_out1.clone().mul(div21_out1);
        let mul14_out1 = constant2_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul13_out1);
        let linear6_out1 = self.linear6.forward(mul14_out1.clone());
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(linear6_out1.clone());
        let mul15_out1 = linear6_out1.mul(sigmoid1_out1);
        let linear7_out1 = self.linear7.forward(mul14_out1);
        let mul16_out1 = mul15_out1.mul(linear7_out1);
        let linear8_out1 = self.linear8.forward(mul16_out1);
        let mul17_out1 = constant4_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear8_out1);
        let add31_out1 = add29_out1.add(mul17_out1);
        let constant509_out1 = 2f32;
        let pow3_out1 = add31_out1.clone().powf_scalar(constant509_out1);
        let reducemean3_out1 = { pow3_out1.mean_dim(2usize) };
        let constant510_out1 = 0.00001f32;
        let add32_out1 = reducemean3_out1.add_scalar(constant510_out1);
        let sqrt3_out1 = add32_out1.sqrt();
        let constant511_out1 = 1f32;
        let div22_out1 = constant511_out1 / sqrt3_out1;
        let mul18_out1 = add31_out1.clone().mul(div22_out1);
        let mul19_out1 = constant5_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul18_out1);
        let shape13_out1: [i64; 3] = {
            let axes = &mul19_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant512_out1 = 0i64;
        let actual_idx = if constant512_out1 < 0 {
            (shape13_out1.len() as i64 + constant512_out1) as usize
        } else {
            constant512_out1 as usize
        };
        let gather29_out1 = shape13_out1[actual_idx] as i64;
        let constant513_out1 = 1i64;
        let actual_idx = if constant513_out1 < 0 {
            (shape13_out1.len() as i64 + constant513_out1) as usize
        } else {
            constant513_out1 as usize
        };
        let gather30_out1 = shape13_out1[actual_idx] as i64;
        let linear9_out1 = self.linear9.forward(mul19_out1.clone());
        let unsqueeze42_out1 = [gather29_out1];
        let unsqueeze43_out1 = [gather30_out1];
        let constant516_out1: [i64; 1] = [-1i64];
        let constant517_out1: [i64; 1] = [64i64];
        let concat15_out1: [i64; 4usize] = [
            &unsqueeze42_out1[..],
            &unsqueeze43_out1[..],
            &constant516_out1[..],
            &constant517_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze44_out1 = [gather29_out1];
        let unsqueeze45_out1 = [gather30_out1];
        let constant520_out1: [i64; 1] = [-1i64];
        let constant521_out1: [i64; 1] = [64i64];
        let concat16_out1: [i64; 4usize] = [
            &unsqueeze44_out1[..],
            &unsqueeze45_out1[..],
            &constant520_out1[..],
            &constant521_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze46_out1 = [gather29_out1];
        let unsqueeze47_out1 = [gather30_out1];
        let constant524_out1: [i64; 1] = [-1i64];
        let constant525_out1: [i64; 1] = [64i64];
        let concat17_out1: [i64; 4usize] = [
            &unsqueeze46_out1[..],
            &unsqueeze47_out1[..],
            &constant524_out1[..],
            &constant525_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape8_out1 = linear9_out1.reshape(concat15_out1);
        let transpose28_out1 = reshape8_out1.permute([0, 2, 1, 3]);
        let linear10_out1 = self.linear10.forward(mul19_out1.clone());
        let reshape9_out1 = linear10_out1.reshape(concat16_out1);
        let transpose29_out1 = reshape9_out1.permute([0, 2, 1, 3]);
        let linear11_out1 = self.linear11.forward(mul19_out1);
        let reshape10_out1 = linear11_out1.reshape(concat17_out1);
        let transpose30_out1 = reshape10_out1.permute([0, 2, 1, 3]);
        let mul20_out1 = transpose28_out1.clone().mul(unsqueeze33_out1.clone());
        let shape15_out1: [i64; 4] = {
            let axes = &transpose28_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant526_out1 = 3i64;
        let actual_idx = if constant526_out1 < 0 {
            (shape15_out1.len() as i64 + constant526_out1) as usize
        } else {
            constant526_out1 as usize
        };
        let gather31_out1 = shape15_out1[actual_idx] as i64;
        let constant527_out1 = 2i64;
        let div23_out1 = gather31_out1 / constant527_out1;
        let unsqueeze48_out1 = [div23_out1];
        let slice28_out1 = transpose28_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze48_out1[0]]);
        let unsqueeze49_out1 = [div23_out1];
        let slice29_out1 = transpose28_out1
            .slice(s![.., .., .., unsqueeze49_out1[0]..9223372036854775807]);
        let neg3_out1 = slice29_out1.neg();
        let concat18_out1 = burn::tensor::Tensor::cat(
            [neg3_out1, slice28_out1].into(),
            3,
        );
        let mul21_out1 = concat18_out1.mul(unsqueeze34_out1.clone());
        let add33_out1 = mul20_out1.add(mul21_out1);
        let mul22_out1 = transpose29_out1.clone().mul(unsqueeze33_out1.clone());
        let shape16_out1: [i64; 4] = {
            let axes = &transpose29_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant536_out1 = 3i64;
        let actual_idx = if constant536_out1 < 0 {
            (shape16_out1.len() as i64 + constant536_out1) as usize
        } else {
            constant536_out1 as usize
        };
        let gather32_out1 = shape16_out1[actual_idx] as i64;
        let constant537_out1 = 2i64;
        let div24_out1 = gather32_out1 / constant537_out1;
        let unsqueeze50_out1 = [div24_out1];
        let slice30_out1 = transpose29_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze50_out1[0]]);
        let unsqueeze51_out1 = [div24_out1];
        let slice31_out1 = transpose29_out1
            .slice(s![.., .., .., unsqueeze51_out1[0]..9223372036854775807]);
        let neg4_out1 = slice31_out1.neg();
        let concat19_out1 = burn::tensor::Tensor::cat(
            [neg4_out1, slice30_out1].into(),
            3,
        );
        let mul23_out1 = concat19_out1.mul(unsqueeze34_out1.clone());
        let add34_out1 = mul22_out1.add(mul23_out1);
        let concat20_out1 = burn::tensor::Tensor::cat(
            [past_key_1, add34_out1].into(),
            2,
        );
        let concat21_out1 = burn::tensor::Tensor::cat(
            [past_value_1, transpose30_out1].into(),
            2,
        );
        let slice32_out1 = concat20_out1.slice(s![.., .., - 72.., ..]);
        let slice33_out1 = concat21_out1.slice(s![.., .., - 72.., ..]);
        let shape17_out1: [i64; 4] = {
            let axes = &slice32_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant554_out1 = 2i64;
        let actual_idx = if constant554_out1 < 0 {
            (shape17_out1.len() as i64 + constant554_out1) as usize
        } else {
            constant554_out1 as usize
        };
        let gather33_out1 = shape17_out1[actual_idx] as i64;
        let unsqueeze52_out1 = [gather33_out1];
        let slice34_out1 = unsqueeze26_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze52_out1[0]]);
        let transpose31_out1 = slice32_out1.clone().permute([0, 1, 3, 2]);
        let constant559_out1 = 0.35355338f32;
        let mul24_out1 = add33_out1.mul_scalar(constant559_out1);
        let constant560_out1 = 0.35355338f32;
        let mul25_out1 = transpose31_out1.mul_scalar(constant560_out1);
        let matmul15_out1 = mul24_out1.matmul(mul25_out1);
        let add35_out1 = matmul15_out1.add(slice34_out1);
        let softmax2_out1 = burn::tensor::activation::softmax(add35_out1, 3);
        let matmul16_out1 = softmax2_out1.matmul(slice33_out1.clone());
        let transpose32_out1 = matmul16_out1.permute([0, 2, 1, 3]);
        let unsqueeze53_out1 = [gather29_out1];
        let unsqueeze54_out1 = [gather30_out1];
        let constant563_out1: [i64; 1] = [-1i64];
        let concat22_out1: [i64; 3usize] = [
            &unsqueeze53_out1[..],
            &unsqueeze54_out1[..],
            &constant563_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape11_out1 = transpose32_out1.reshape(concat22_out1);
        let linear12_out1 = self.linear12.forward(reshape11_out1);
        let mul26_out1 = constant7_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear12_out1);
        let add36_out1 = add31_out1.add(mul26_out1);
        let constant564_out1 = 2f32;
        let pow4_out1 = add36_out1.clone().powf_scalar(constant564_out1);
        let reducemean4_out1 = { pow4_out1.mean_dim(2usize) };
        let constant565_out1 = 0.00001f32;
        let add37_out1 = reducemean4_out1.add_scalar(constant565_out1);
        let sqrt4_out1 = add37_out1.sqrt();
        let constant566_out1 = 1f32;
        let div25_out1 = constant566_out1 / sqrt4_out1;
        let mul27_out1 = add36_out1.clone().mul(div25_out1);
        let mul28_out1 = constant6_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul27_out1);
        let linear13_out1 = self.linear13.forward(mul28_out1.clone());
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(linear13_out1.clone());
        let mul29_out1 = linear13_out1.mul(sigmoid2_out1);
        let linear14_out1 = self.linear14.forward(mul28_out1);
        let mul30_out1 = mul29_out1.mul(linear14_out1);
        let linear15_out1 = self.linear15.forward(mul30_out1);
        let mul31_out1 = constant8_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear15_out1);
        let add38_out1 = add36_out1.add(mul31_out1);
        let constant567_out1 = 2f32;
        let pow5_out1 = add38_out1.clone().powf_scalar(constant567_out1);
        let reducemean5_out1 = { pow5_out1.mean_dim(2usize) };
        let constant568_out1 = 0.00001f32;
        let add39_out1 = reducemean5_out1.add_scalar(constant568_out1);
        let sqrt5_out1 = add39_out1.sqrt();
        let constant569_out1 = 1f32;
        let div26_out1 = constant569_out1 / sqrt5_out1;
        let mul32_out1 = add38_out1.clone().mul(div26_out1);
        let mul33_out1 = constant9_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul32_out1);
        let shape18_out1: [i64; 3] = {
            let axes = &mul33_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant570_out1 = 0i64;
        let actual_idx = if constant570_out1 < 0 {
            (shape18_out1.len() as i64 + constant570_out1) as usize
        } else {
            constant570_out1 as usize
        };
        let gather34_out1 = shape18_out1[actual_idx] as i64;
        let constant571_out1 = 1i64;
        let actual_idx = if constant571_out1 < 0 {
            (shape18_out1.len() as i64 + constant571_out1) as usize
        } else {
            constant571_out1 as usize
        };
        let gather35_out1 = shape18_out1[actual_idx] as i64;
        let linear16_out1 = self.linear16.forward(mul33_out1.clone());
        let unsqueeze55_out1 = [gather34_out1];
        let unsqueeze56_out1 = [gather35_out1];
        let constant574_out1: [i64; 1] = [-1i64];
        let constant575_out1: [i64; 1] = [64i64];
        let concat23_out1: [i64; 4usize] = [
            &unsqueeze55_out1[..],
            &unsqueeze56_out1[..],
            &constant574_out1[..],
            &constant575_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze57_out1 = [gather34_out1];
        let unsqueeze58_out1 = [gather35_out1];
        let constant578_out1: [i64; 1] = [-1i64];
        let constant579_out1: [i64; 1] = [64i64];
        let concat24_out1: [i64; 4usize] = [
            &unsqueeze57_out1[..],
            &unsqueeze58_out1[..],
            &constant578_out1[..],
            &constant579_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze59_out1 = [gather34_out1];
        let unsqueeze60_out1 = [gather35_out1];
        let constant582_out1: [i64; 1] = [-1i64];
        let constant583_out1: [i64; 1] = [64i64];
        let concat25_out1: [i64; 4usize] = [
            &unsqueeze59_out1[..],
            &unsqueeze60_out1[..],
            &constant582_out1[..],
            &constant583_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape12_out1 = linear16_out1.reshape(concat23_out1);
        let transpose33_out1 = reshape12_out1.permute([0, 2, 1, 3]);
        let linear17_out1 = self.linear17.forward(mul33_out1.clone());
        let reshape13_out1 = linear17_out1.reshape(concat24_out1);
        let transpose34_out1 = reshape13_out1.permute([0, 2, 1, 3]);
        let linear18_out1 = self.linear18.forward(mul33_out1);
        let reshape14_out1 = linear18_out1.reshape(concat25_out1);
        let transpose35_out1 = reshape14_out1.permute([0, 2, 1, 3]);
        let mul34_out1 = transpose33_out1.clone().mul(unsqueeze33_out1.clone());
        let shape20_out1: [i64; 4] = {
            let axes = &transpose33_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant584_out1 = 3i64;
        let actual_idx = if constant584_out1 < 0 {
            (shape20_out1.len() as i64 + constant584_out1) as usize
        } else {
            constant584_out1 as usize
        };
        let gather36_out1 = shape20_out1[actual_idx] as i64;
        let constant585_out1 = 2i64;
        let div27_out1 = gather36_out1 / constant585_out1;
        let unsqueeze61_out1 = [div27_out1];
        let slice35_out1 = transpose33_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze61_out1[0]]);
        let unsqueeze62_out1 = [div27_out1];
        let slice36_out1 = transpose33_out1
            .slice(s![.., .., .., unsqueeze62_out1[0]..9223372036854775807]);
        let neg5_out1 = slice36_out1.neg();
        let concat26_out1 = burn::tensor::Tensor::cat(
            [neg5_out1, slice35_out1].into(),
            3,
        );
        let mul35_out1 = concat26_out1.mul(unsqueeze34_out1.clone());
        let add40_out1 = mul34_out1.add(mul35_out1);
        let mul36_out1 = transpose34_out1.clone().mul(unsqueeze33_out1.clone());
        let shape21_out1: [i64; 4] = {
            let axes = &transpose34_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant594_out1 = 3i64;
        let actual_idx = if constant594_out1 < 0 {
            (shape21_out1.len() as i64 + constant594_out1) as usize
        } else {
            constant594_out1 as usize
        };
        let gather37_out1 = shape21_out1[actual_idx] as i64;
        let constant595_out1 = 2i64;
        let div28_out1 = gather37_out1 / constant595_out1;
        let unsqueeze63_out1 = [div28_out1];
        let slice37_out1 = transpose34_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze63_out1[0]]);
        let unsqueeze64_out1 = [div28_out1];
        let slice38_out1 = transpose34_out1
            .slice(s![.., .., .., unsqueeze64_out1[0]..9223372036854775807]);
        let neg6_out1 = slice38_out1.neg();
        let concat27_out1 = burn::tensor::Tensor::cat(
            [neg6_out1, slice37_out1].into(),
            3,
        );
        let mul37_out1 = concat27_out1.mul(unsqueeze34_out1.clone());
        let add41_out1 = mul36_out1.add(mul37_out1);
        let concat28_out1 = burn::tensor::Tensor::cat(
            [past_key_2, add41_out1].into(),
            2,
        );
        let concat29_out1 = burn::tensor::Tensor::cat(
            [past_value_2, transpose35_out1].into(),
            2,
        );
        let slice39_out1 = concat28_out1.slice(s![.., .., - 72.., ..]);
        let slice40_out1 = concat29_out1.slice(s![.., .., - 72.., ..]);
        let shape22_out1: [i64; 4] = {
            let axes = &slice39_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant612_out1 = 2i64;
        let actual_idx = if constant612_out1 < 0 {
            (shape22_out1.len() as i64 + constant612_out1) as usize
        } else {
            constant612_out1 as usize
        };
        let gather38_out1 = shape22_out1[actual_idx] as i64;
        let unsqueeze65_out1 = [gather38_out1];
        let slice41_out1 = unsqueeze26_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze65_out1[0]]);
        let transpose36_out1 = slice39_out1.clone().permute([0, 1, 3, 2]);
        let constant617_out1 = 0.35355338f32;
        let mul38_out1 = add40_out1.mul_scalar(constant617_out1);
        let constant618_out1 = 0.35355338f32;
        let mul39_out1 = transpose36_out1.mul_scalar(constant618_out1);
        let matmul24_out1 = mul38_out1.matmul(mul39_out1);
        let add42_out1 = matmul24_out1.add(slice41_out1);
        let softmax3_out1 = burn::tensor::activation::softmax(add42_out1, 3);
        let matmul25_out1 = softmax3_out1.matmul(slice40_out1.clone());
        let transpose37_out1 = matmul25_out1.permute([0, 2, 1, 3]);
        let unsqueeze66_out1 = [gather34_out1];
        let unsqueeze67_out1 = [gather35_out1];
        let constant621_out1: [i64; 1] = [-1i64];
        let concat30_out1: [i64; 3usize] = [
            &unsqueeze66_out1[..],
            &unsqueeze67_out1[..],
            &constant621_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape15_out1 = transpose37_out1.reshape(concat30_out1);
        let linear19_out1 = self.linear19.forward(reshape15_out1);
        let mul40_out1 = constant11_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear19_out1);
        let add43_out1 = add38_out1.add(mul40_out1);
        let constant622_out1 = 2f32;
        let pow6_out1 = add43_out1.clone().powf_scalar(constant622_out1);
        let reducemean6_out1 = { pow6_out1.mean_dim(2usize) };
        let constant623_out1 = 0.00001f32;
        let add44_out1 = reducemean6_out1.add_scalar(constant623_out1);
        let sqrt6_out1 = add44_out1.sqrt();
        let constant624_out1 = 1f32;
        let div29_out1 = constant624_out1 / sqrt6_out1;
        let mul41_out1 = add43_out1.clone().mul(div29_out1);
        let mul42_out1 = constant10_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul41_out1);
        let linear20_out1 = self.linear20.forward(mul42_out1.clone());
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(linear20_out1.clone());
        let mul43_out1 = linear20_out1.mul(sigmoid3_out1);
        let linear21_out1 = self.linear21.forward(mul42_out1);
        let mul44_out1 = mul43_out1.mul(linear21_out1);
        let linear22_out1 = self.linear22.forward(mul44_out1);
        let mul45_out1 = constant12_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear22_out1);
        let add45_out1 = add43_out1.add(mul45_out1);
        let constant625_out1 = 2f32;
        let pow7_out1 = add45_out1.clone().powf_scalar(constant625_out1);
        let reducemean7_out1 = { pow7_out1.mean_dim(2usize) };
        let constant626_out1 = 0.00001f32;
        let add46_out1 = reducemean7_out1.add_scalar(constant626_out1);
        let sqrt7_out1 = add46_out1.sqrt();
        let constant627_out1 = 1f32;
        let div30_out1 = constant627_out1 / sqrt7_out1;
        let mul46_out1 = add45_out1.clone().mul(div30_out1);
        let mul47_out1 = constant13_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul46_out1);
        let shape23_out1: [i64; 3] = {
            let axes = &mul47_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant628_out1 = 0i64;
        let actual_idx = if constant628_out1 < 0 {
            (shape23_out1.len() as i64 + constant628_out1) as usize
        } else {
            constant628_out1 as usize
        };
        let gather39_out1 = shape23_out1[actual_idx] as i64;
        let constant629_out1 = 1i64;
        let actual_idx = if constant629_out1 < 0 {
            (shape23_out1.len() as i64 + constant629_out1) as usize
        } else {
            constant629_out1 as usize
        };
        let gather40_out1 = shape23_out1[actual_idx] as i64;
        let linear23_out1 = self.linear23.forward(mul47_out1.clone());
        let unsqueeze68_out1 = [gather39_out1];
        let unsqueeze69_out1 = [gather40_out1];
        let constant632_out1: [i64; 1] = [-1i64];
        let constant633_out1: [i64; 1] = [64i64];
        let concat31_out1: [i64; 4usize] = [
            &unsqueeze68_out1[..],
            &unsqueeze69_out1[..],
            &constant632_out1[..],
            &constant633_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze70_out1 = [gather39_out1];
        let unsqueeze71_out1 = [gather40_out1];
        let constant636_out1: [i64; 1] = [-1i64];
        let constant637_out1: [i64; 1] = [64i64];
        let concat32_out1: [i64; 4usize] = [
            &unsqueeze70_out1[..],
            &unsqueeze71_out1[..],
            &constant636_out1[..],
            &constant637_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze72_out1 = [gather39_out1];
        let unsqueeze73_out1 = [gather40_out1];
        let constant640_out1: [i64; 1] = [-1i64];
        let constant641_out1: [i64; 1] = [64i64];
        let concat33_out1: [i64; 4usize] = [
            &unsqueeze72_out1[..],
            &unsqueeze73_out1[..],
            &constant640_out1[..],
            &constant641_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape16_out1 = linear23_out1.reshape(concat31_out1);
        let transpose38_out1 = reshape16_out1.permute([0, 2, 1, 3]);
        let linear24_out1 = self.linear24.forward(mul47_out1.clone());
        let reshape17_out1 = linear24_out1.reshape(concat32_out1);
        let transpose39_out1 = reshape17_out1.permute([0, 2, 1, 3]);
        let linear25_out1 = self.linear25.forward(mul47_out1);
        let reshape18_out1 = linear25_out1.reshape(concat33_out1);
        let transpose40_out1 = reshape18_out1.permute([0, 2, 1, 3]);
        let mul48_out1 = transpose38_out1.clone().mul(unsqueeze33_out1.clone());
        let shape25_out1: [i64; 4] = {
            let axes = &transpose38_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant642_out1 = 3i64;
        let actual_idx = if constant642_out1 < 0 {
            (shape25_out1.len() as i64 + constant642_out1) as usize
        } else {
            constant642_out1 as usize
        };
        let gather41_out1 = shape25_out1[actual_idx] as i64;
        let constant643_out1 = 2i64;
        let div31_out1 = gather41_out1 / constant643_out1;
        let unsqueeze74_out1 = [div31_out1];
        let slice42_out1 = transpose38_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze74_out1[0]]);
        let unsqueeze75_out1 = [div31_out1];
        let slice43_out1 = transpose38_out1
            .slice(s![.., .., .., unsqueeze75_out1[0]..9223372036854775807]);
        let neg7_out1 = slice43_out1.neg();
        let concat34_out1 = burn::tensor::Tensor::cat(
            [neg7_out1, slice42_out1].into(),
            3,
        );
        let mul49_out1 = concat34_out1.mul(unsqueeze34_out1.clone());
        let add47_out1 = mul48_out1.add(mul49_out1);
        let mul50_out1 = transpose39_out1.clone().mul(unsqueeze33_out1.clone());
        let shape26_out1: [i64; 4] = {
            let axes = &transpose39_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant652_out1 = 3i64;
        let actual_idx = if constant652_out1 < 0 {
            (shape26_out1.len() as i64 + constant652_out1) as usize
        } else {
            constant652_out1 as usize
        };
        let gather42_out1 = shape26_out1[actual_idx] as i64;
        let constant653_out1 = 2i64;
        let div32_out1 = gather42_out1 / constant653_out1;
        let unsqueeze76_out1 = [div32_out1];
        let slice44_out1 = transpose39_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze76_out1[0]]);
        let unsqueeze77_out1 = [div32_out1];
        let slice45_out1 = transpose39_out1
            .slice(s![.., .., .., unsqueeze77_out1[0]..9223372036854775807]);
        let neg8_out1 = slice45_out1.neg();
        let concat35_out1 = burn::tensor::Tensor::cat(
            [neg8_out1, slice44_out1].into(),
            3,
        );
        let mul51_out1 = concat35_out1.mul(unsqueeze34_out1.clone());
        let add48_out1 = mul50_out1.add(mul51_out1);
        let concat36_out1 = burn::tensor::Tensor::cat(
            [past_key_3, add48_out1].into(),
            2,
        );
        let concat37_out1 = burn::tensor::Tensor::cat(
            [past_value_3, transpose40_out1].into(),
            2,
        );
        let slice46_out1 = concat36_out1.slice(s![.., .., - 72.., ..]);
        let slice47_out1 = concat37_out1.slice(s![.., .., - 72.., ..]);
        let shape27_out1: [i64; 4] = {
            let axes = &slice46_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant670_out1 = 2i64;
        let actual_idx = if constant670_out1 < 0 {
            (shape27_out1.len() as i64 + constant670_out1) as usize
        } else {
            constant670_out1 as usize
        };
        let gather43_out1 = shape27_out1[actual_idx] as i64;
        let unsqueeze78_out1 = [gather43_out1];
        let slice48_out1 = unsqueeze26_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze78_out1[0]]);
        let transpose41_out1 = slice46_out1.clone().permute([0, 1, 3, 2]);
        let constant675_out1 = 0.35355338f32;
        let mul52_out1 = add47_out1.mul_scalar(constant675_out1);
        let constant676_out1 = 0.35355338f32;
        let mul53_out1 = transpose41_out1.mul_scalar(constant676_out1);
        let matmul33_out1 = mul52_out1.matmul(mul53_out1);
        let add49_out1 = matmul33_out1.add(slice48_out1);
        let softmax4_out1 = burn::tensor::activation::softmax(add49_out1, 3);
        let matmul34_out1 = softmax4_out1.matmul(slice47_out1.clone());
        let transpose42_out1 = matmul34_out1.permute([0, 2, 1, 3]);
        let unsqueeze79_out1 = [gather39_out1];
        let unsqueeze80_out1 = [gather40_out1];
        let constant679_out1: [i64; 1] = [-1i64];
        let concat38_out1: [i64; 3usize] = [
            &unsqueeze79_out1[..],
            &unsqueeze80_out1[..],
            &constant679_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape19_out1 = transpose42_out1.reshape(concat38_out1);
        let linear26_out1 = self.linear26.forward(reshape19_out1);
        let mul54_out1 = constant15_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear26_out1);
        let add50_out1 = add45_out1.add(mul54_out1);
        let constant680_out1 = 2f32;
        let pow8_out1 = add50_out1.clone().powf_scalar(constant680_out1);
        let reducemean8_out1 = { pow8_out1.mean_dim(2usize) };
        let constant681_out1 = 0.00001f32;
        let add51_out1 = reducemean8_out1.add_scalar(constant681_out1);
        let sqrt8_out1 = add51_out1.sqrt();
        let constant682_out1 = 1f32;
        let div33_out1 = constant682_out1 / sqrt8_out1;
        let mul55_out1 = add50_out1.clone().mul(div33_out1);
        let mul56_out1 = constant14_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul55_out1);
        let linear27_out1 = self.linear27.forward(mul56_out1.clone());
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(linear27_out1.clone());
        let mul57_out1 = linear27_out1.mul(sigmoid4_out1);
        let linear28_out1 = self.linear28.forward(mul56_out1);
        let mul58_out1 = mul57_out1.mul(linear28_out1);
        let linear29_out1 = self.linear29.forward(mul58_out1);
        let mul59_out1 = constant16_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear29_out1);
        let add52_out1 = add50_out1.add(mul59_out1);
        let constant683_out1 = 2f32;
        let pow9_out1 = add52_out1.clone().powf_scalar(constant683_out1);
        let reducemean9_out1 = { pow9_out1.mean_dim(2usize) };
        let constant684_out1 = 0.00001f32;
        let add53_out1 = reducemean9_out1.add_scalar(constant684_out1);
        let sqrt9_out1 = add53_out1.sqrt();
        let constant685_out1 = 1f32;
        let div34_out1 = constant685_out1 / sqrt9_out1;
        let mul60_out1 = add52_out1.clone().mul(div34_out1);
        let mul61_out1 = constant17_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul60_out1);
        let shape28_out1: [i64; 3] = {
            let axes = &mul61_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant686_out1 = 0i64;
        let actual_idx = if constant686_out1 < 0 {
            (shape28_out1.len() as i64 + constant686_out1) as usize
        } else {
            constant686_out1 as usize
        };
        let gather44_out1 = shape28_out1[actual_idx] as i64;
        let constant687_out1 = 1i64;
        let actual_idx = if constant687_out1 < 0 {
            (shape28_out1.len() as i64 + constant687_out1) as usize
        } else {
            constant687_out1 as usize
        };
        let gather45_out1 = shape28_out1[actual_idx] as i64;
        let linear30_out1 = self.linear30.forward(mul61_out1.clone());
        let unsqueeze81_out1 = [gather44_out1];
        let unsqueeze82_out1 = [gather45_out1];
        let constant690_out1: [i64; 1] = [-1i64];
        let constant691_out1: [i64; 1] = [64i64];
        let concat39_out1: [i64; 4usize] = [
            &unsqueeze81_out1[..],
            &unsqueeze82_out1[..],
            &constant690_out1[..],
            &constant691_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze83_out1 = [gather44_out1];
        let unsqueeze84_out1 = [gather45_out1];
        let constant694_out1: [i64; 1] = [-1i64];
        let constant695_out1: [i64; 1] = [64i64];
        let concat40_out1: [i64; 4usize] = [
            &unsqueeze83_out1[..],
            &unsqueeze84_out1[..],
            &constant694_out1[..],
            &constant695_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze85_out1 = [gather44_out1];
        let unsqueeze86_out1 = [gather45_out1];
        let constant698_out1: [i64; 1] = [-1i64];
        let constant699_out1: [i64; 1] = [64i64];
        let concat41_out1: [i64; 4usize] = [
            &unsqueeze85_out1[..],
            &unsqueeze86_out1[..],
            &constant698_out1[..],
            &constant699_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape20_out1 = linear30_out1.reshape(concat39_out1);
        let transpose43_out1 = reshape20_out1.permute([0, 2, 1, 3]);
        let linear31_out1 = self.linear31.forward(mul61_out1.clone());
        let reshape21_out1 = linear31_out1.reshape(concat40_out1);
        let transpose44_out1 = reshape21_out1.permute([0, 2, 1, 3]);
        let linear32_out1 = self.linear32.forward(mul61_out1);
        let reshape22_out1 = linear32_out1.reshape(concat41_out1);
        let transpose45_out1 = reshape22_out1.permute([0, 2, 1, 3]);
        let mul62_out1 = transpose43_out1.clone().mul(unsqueeze33_out1.clone());
        let shape30_out1: [i64; 4] = {
            let axes = &transpose43_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant700_out1 = 3i64;
        let actual_idx = if constant700_out1 < 0 {
            (shape30_out1.len() as i64 + constant700_out1) as usize
        } else {
            constant700_out1 as usize
        };
        let gather46_out1 = shape30_out1[actual_idx] as i64;
        let constant701_out1 = 2i64;
        let div35_out1 = gather46_out1 / constant701_out1;
        let unsqueeze87_out1 = [div35_out1];
        let slice49_out1 = transpose43_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze87_out1[0]]);
        let unsqueeze88_out1 = [div35_out1];
        let slice50_out1 = transpose43_out1
            .slice(s![.., .., .., unsqueeze88_out1[0]..9223372036854775807]);
        let neg9_out1 = slice50_out1.neg();
        let concat42_out1 = burn::tensor::Tensor::cat(
            [neg9_out1, slice49_out1].into(),
            3,
        );
        let mul63_out1 = concat42_out1.mul(unsqueeze34_out1.clone());
        let add54_out1 = mul62_out1.add(mul63_out1);
        let mul64_out1 = transpose44_out1.clone().mul(unsqueeze33_out1.clone());
        let shape31_out1: [i64; 4] = {
            let axes = &transpose44_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant710_out1 = 3i64;
        let actual_idx = if constant710_out1 < 0 {
            (shape31_out1.len() as i64 + constant710_out1) as usize
        } else {
            constant710_out1 as usize
        };
        let gather47_out1 = shape31_out1[actual_idx] as i64;
        let constant711_out1 = 2i64;
        let div36_out1 = gather47_out1 / constant711_out1;
        let unsqueeze89_out1 = [div36_out1];
        let slice51_out1 = transpose44_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze89_out1[0]]);
        let unsqueeze90_out1 = [div36_out1];
        let slice52_out1 = transpose44_out1
            .slice(s![.., .., .., unsqueeze90_out1[0]..9223372036854775807]);
        let neg10_out1 = slice52_out1.neg();
        let concat43_out1 = burn::tensor::Tensor::cat(
            [neg10_out1, slice51_out1].into(),
            3,
        );
        let mul65_out1 = concat43_out1.mul(unsqueeze34_out1.clone());
        let add55_out1 = mul64_out1.add(mul65_out1);
        let concat44_out1 = burn::tensor::Tensor::cat(
            [past_key_4, add55_out1].into(),
            2,
        );
        let concat45_out1 = burn::tensor::Tensor::cat(
            [past_value_4, transpose45_out1].into(),
            2,
        );
        let slice53_out1 = concat44_out1.slice(s![.., .., - 72.., ..]);
        let slice54_out1 = concat45_out1.slice(s![.., .., - 72.., ..]);
        let shape32_out1: [i64; 4] = {
            let axes = &slice53_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant728_out1 = 2i64;
        let actual_idx = if constant728_out1 < 0 {
            (shape32_out1.len() as i64 + constant728_out1) as usize
        } else {
            constant728_out1 as usize
        };
        let gather48_out1 = shape32_out1[actual_idx] as i64;
        let unsqueeze91_out1 = [gather48_out1];
        let slice55_out1 = unsqueeze26_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze91_out1[0]]);
        let transpose46_out1 = slice53_out1.clone().permute([0, 1, 3, 2]);
        let constant733_out1 = 0.35355338f32;
        let mul66_out1 = add54_out1.mul_scalar(constant733_out1);
        let constant734_out1 = 0.35355338f32;
        let mul67_out1 = transpose46_out1.mul_scalar(constant734_out1);
        let matmul42_out1 = mul66_out1.matmul(mul67_out1);
        let add56_out1 = matmul42_out1.add(slice55_out1);
        let softmax5_out1 = burn::tensor::activation::softmax(add56_out1, 3);
        let matmul43_out1 = softmax5_out1.matmul(slice54_out1.clone());
        let transpose47_out1 = matmul43_out1.permute([0, 2, 1, 3]);
        let unsqueeze92_out1 = [gather44_out1];
        let unsqueeze93_out1 = [gather45_out1];
        let constant737_out1: [i64; 1] = [-1i64];
        let concat46_out1: [i64; 3usize] = [
            &unsqueeze92_out1[..],
            &unsqueeze93_out1[..],
            &constant737_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape23_out1 = transpose47_out1.reshape(concat46_out1);
        let linear33_out1 = self.linear33.forward(reshape23_out1);
        let mul68_out1 = constant19_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear33_out1);
        let add57_out1 = add52_out1.add(mul68_out1);
        let constant738_out1 = 2f32;
        let pow10_out1 = add57_out1.clone().powf_scalar(constant738_out1);
        let reducemean10_out1 = { pow10_out1.mean_dim(2usize) };
        let constant739_out1 = 0.00001f32;
        let add58_out1 = reducemean10_out1.add_scalar(constant739_out1);
        let sqrt10_out1 = add58_out1.sqrt();
        let constant740_out1 = 1f32;
        let div37_out1 = constant740_out1 / sqrt10_out1;
        let mul69_out1 = add57_out1.clone().mul(div37_out1);
        let mul70_out1 = constant18_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul69_out1);
        let linear34_out1 = self.linear34.forward(mul70_out1.clone());
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(linear34_out1.clone());
        let mul71_out1 = linear34_out1.mul(sigmoid5_out1);
        let linear35_out1 = self.linear35.forward(mul70_out1);
        let mul72_out1 = mul71_out1.mul(linear35_out1);
        let linear36_out1 = self.linear36.forward(mul72_out1);
        let mul73_out1 = constant20_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear36_out1);
        let add59_out1 = add57_out1.add(mul73_out1);
        let constant741_out1 = 2f32;
        let pow11_out1 = add59_out1.clone().powf_scalar(constant741_out1);
        let reducemean11_out1 = { pow11_out1.mean_dim(2usize) };
        let constant742_out1 = 0.00001f32;
        let add60_out1 = reducemean11_out1.add_scalar(constant742_out1);
        let sqrt11_out1 = add60_out1.sqrt();
        let constant743_out1 = 1f32;
        let div38_out1 = constant743_out1 / sqrt11_out1;
        let mul74_out1 = add59_out1.clone().mul(div38_out1);
        let mul75_out1 = constant21_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul74_out1);
        let shape33_out1: [i64; 3] = {
            let axes = &mul75_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant744_out1 = 0i64;
        let actual_idx = if constant744_out1 < 0 {
            (shape33_out1.len() as i64 + constant744_out1) as usize
        } else {
            constant744_out1 as usize
        };
        let gather49_out1 = shape33_out1[actual_idx] as i64;
        let constant745_out1 = 1i64;
        let actual_idx = if constant745_out1 < 0 {
            (shape33_out1.len() as i64 + constant745_out1) as usize
        } else {
            constant745_out1 as usize
        };
        let gather50_out1 = shape33_out1[actual_idx] as i64;
        let linear37_out1 = self.linear37.forward(mul75_out1.clone());
        let unsqueeze94_out1 = [gather49_out1];
        let unsqueeze95_out1 = [gather50_out1];
        let constant748_out1: [i64; 1] = [-1i64];
        let constant749_out1: [i64; 1] = [64i64];
        let concat47_out1: [i64; 4usize] = [
            &unsqueeze94_out1[..],
            &unsqueeze95_out1[..],
            &constant748_out1[..],
            &constant749_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze96_out1 = [gather49_out1];
        let unsqueeze97_out1 = [gather50_out1];
        let constant752_out1: [i64; 1] = [-1i64];
        let constant753_out1: [i64; 1] = [64i64];
        let concat48_out1: [i64; 4usize] = [
            &unsqueeze96_out1[..],
            &unsqueeze97_out1[..],
            &constant752_out1[..],
            &constant753_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze98_out1 = [gather49_out1];
        let unsqueeze99_out1 = [gather50_out1];
        let constant756_out1: [i64; 1] = [-1i64];
        let constant757_out1: [i64; 1] = [64i64];
        let concat49_out1: [i64; 4usize] = [
            &unsqueeze98_out1[..],
            &unsqueeze99_out1[..],
            &constant756_out1[..],
            &constant757_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape24_out1 = linear37_out1.reshape(concat47_out1);
        let transpose48_out1 = reshape24_out1.permute([0, 2, 1, 3]);
        let linear38_out1 = self.linear38.forward(mul75_out1.clone());
        let reshape25_out1 = linear38_out1.reshape(concat48_out1);
        let transpose49_out1 = reshape25_out1.permute([0, 2, 1, 3]);
        let linear39_out1 = self.linear39.forward(mul75_out1);
        let reshape26_out1 = linear39_out1.reshape(concat49_out1);
        let transpose50_out1 = reshape26_out1.permute([0, 2, 1, 3]);
        let mul76_out1 = transpose48_out1.clone().mul(unsqueeze33_out1.clone());
        let shape35_out1: [i64; 4] = {
            let axes = &transpose48_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant758_out1 = 3i64;
        let actual_idx = if constant758_out1 < 0 {
            (shape35_out1.len() as i64 + constant758_out1) as usize
        } else {
            constant758_out1 as usize
        };
        let gather51_out1 = shape35_out1[actual_idx] as i64;
        let constant759_out1 = 2i64;
        let div39_out1 = gather51_out1 / constant759_out1;
        let unsqueeze100_out1 = [div39_out1];
        let slice56_out1 = transpose48_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze100_out1[0]]);
        let unsqueeze101_out1 = [div39_out1];
        let slice57_out1 = transpose48_out1
            .slice(s![.., .., .., unsqueeze101_out1[0]..9223372036854775807]);
        let neg11_out1 = slice57_out1.neg();
        let concat50_out1 = burn::tensor::Tensor::cat(
            [neg11_out1, slice56_out1].into(),
            3,
        );
        let mul77_out1 = concat50_out1.mul(unsqueeze34_out1.clone());
        let add61_out1 = mul76_out1.add(mul77_out1);
        let mul78_out1 = transpose49_out1.clone().mul(unsqueeze33_out1.clone());
        let shape36_out1: [i64; 4] = {
            let axes = &transpose49_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant768_out1 = 3i64;
        let actual_idx = if constant768_out1 < 0 {
            (shape36_out1.len() as i64 + constant768_out1) as usize
        } else {
            constant768_out1 as usize
        };
        let gather52_out1 = shape36_out1[actual_idx] as i64;
        let constant769_out1 = 2i64;
        let div40_out1 = gather52_out1 / constant769_out1;
        let unsqueeze102_out1 = [div40_out1];
        let slice58_out1 = transpose49_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze102_out1[0]]);
        let unsqueeze103_out1 = [div40_out1];
        let slice59_out1 = transpose49_out1
            .slice(s![.., .., .., unsqueeze103_out1[0]..9223372036854775807]);
        let neg12_out1 = slice59_out1.neg();
        let concat51_out1 = burn::tensor::Tensor::cat(
            [neg12_out1, slice58_out1].into(),
            3,
        );
        let mul79_out1 = concat51_out1.mul(unsqueeze34_out1.clone());
        let add62_out1 = mul78_out1.add(mul79_out1);
        let concat52_out1 = burn::tensor::Tensor::cat(
            [past_key_5, add62_out1].into(),
            2,
        );
        let concat53_out1 = burn::tensor::Tensor::cat(
            [past_value_5, transpose50_out1].into(),
            2,
        );
        let slice60_out1 = concat52_out1.slice(s![.., .., - 72.., ..]);
        let slice61_out1 = concat53_out1.slice(s![.., .., - 72.., ..]);
        let shape37_out1: [i64; 4] = {
            let axes = &slice60_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant786_out1 = 2i64;
        let actual_idx = if constant786_out1 < 0 {
            (shape37_out1.len() as i64 + constant786_out1) as usize
        } else {
            constant786_out1 as usize
        };
        let gather53_out1 = shape37_out1[actual_idx] as i64;
        let unsqueeze104_out1 = [gather53_out1];
        let slice62_out1 = unsqueeze26_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze104_out1[0]]);
        let transpose51_out1 = slice60_out1.clone().permute([0, 1, 3, 2]);
        let constant791_out1 = 0.35355338f32;
        let mul80_out1 = add61_out1.mul_scalar(constant791_out1);
        let constant792_out1 = 0.35355338f32;
        let mul81_out1 = transpose51_out1.mul_scalar(constant792_out1);
        let matmul51_out1 = mul80_out1.matmul(mul81_out1);
        let add63_out1 = matmul51_out1.add(slice62_out1);
        let softmax6_out1 = burn::tensor::activation::softmax(add63_out1, 3);
        let matmul52_out1 = softmax6_out1.matmul(slice61_out1.clone());
        let transpose52_out1 = matmul52_out1.permute([0, 2, 1, 3]);
        let unsqueeze105_out1 = [gather49_out1];
        let unsqueeze106_out1 = [gather50_out1];
        let constant795_out1: [i64; 1] = [-1i64];
        let concat54_out1: [i64; 3usize] = [
            &unsqueeze105_out1[..],
            &unsqueeze106_out1[..],
            &constant795_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape27_out1 = transpose52_out1.reshape(concat54_out1);
        let linear40_out1 = self.linear40.forward(reshape27_out1);
        let mul82_out1 = constant23_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear40_out1);
        let add64_out1 = add59_out1.add(mul82_out1);
        let constant796_out1 = 2f32;
        let pow12_out1 = add64_out1.clone().powf_scalar(constant796_out1);
        let reducemean12_out1 = { pow12_out1.mean_dim(2usize) };
        let constant797_out1 = 0.00001f32;
        let add65_out1 = reducemean12_out1.add_scalar(constant797_out1);
        let sqrt12_out1 = add65_out1.sqrt();
        let constant798_out1 = 1f32;
        let div41_out1 = constant798_out1 / sqrt12_out1;
        let mul83_out1 = add64_out1.clone().mul(div41_out1);
        let mul84_out1 = constant22_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul83_out1);
        let linear41_out1 = self.linear41.forward(mul84_out1.clone());
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(linear41_out1.clone());
        let mul85_out1 = linear41_out1.mul(sigmoid6_out1);
        let linear42_out1 = self.linear42.forward(mul84_out1);
        let mul86_out1 = mul85_out1.mul(linear42_out1);
        let linear43_out1 = self.linear43.forward(mul86_out1);
        let mul87_out1 = constant24_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear43_out1);
        let add66_out1 = add64_out1.add(mul87_out1);
        let constant799_out1 = 2f32;
        let pow13_out1 = add66_out1.clone().powf_scalar(constant799_out1);
        let reducemean13_out1 = { pow13_out1.mean_dim(2usize) };
        let constant800_out1 = 0.00001f32;
        let add67_out1 = reducemean13_out1.add_scalar(constant800_out1);
        let sqrt13_out1 = add67_out1.sqrt();
        let constant801_out1 = 1f32;
        let div42_out1 = constant801_out1 / sqrt13_out1;
        let mul88_out1 = add66_out1.clone().mul(div42_out1);
        let mul89_out1 = constant25_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul88_out1);
        let shape38_out1: [i64; 3] = {
            let axes = &mul89_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant802_out1 = 0i64;
        let actual_idx = if constant802_out1 < 0 {
            (shape38_out1.len() as i64 + constant802_out1) as usize
        } else {
            constant802_out1 as usize
        };
        let gather54_out1 = shape38_out1[actual_idx] as i64;
        let constant803_out1 = 1i64;
        let actual_idx = if constant803_out1 < 0 {
            (shape38_out1.len() as i64 + constant803_out1) as usize
        } else {
            constant803_out1 as usize
        };
        let gather55_out1 = shape38_out1[actual_idx] as i64;
        let linear44_out1 = self.linear44.forward(mul89_out1.clone());
        let unsqueeze107_out1 = [gather54_out1];
        let unsqueeze108_out1 = [gather55_out1];
        let constant806_out1: [i64; 1] = [-1i64];
        let constant807_out1: [i64; 1] = [64i64];
        let concat55_out1: [i64; 4usize] = [
            &unsqueeze107_out1[..],
            &unsqueeze108_out1[..],
            &constant806_out1[..],
            &constant807_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze109_out1 = [gather54_out1];
        let unsqueeze110_out1 = [gather55_out1];
        let constant810_out1: [i64; 1] = [-1i64];
        let constant811_out1: [i64; 1] = [64i64];
        let concat56_out1: [i64; 4usize] = [
            &unsqueeze109_out1[..],
            &unsqueeze110_out1[..],
            &constant810_out1[..],
            &constant811_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze111_out1 = [gather54_out1];
        let unsqueeze112_out1 = [gather55_out1];
        let constant814_out1: [i64; 1] = [-1i64];
        let constant815_out1: [i64; 1] = [64i64];
        let concat57_out1: [i64; 4usize] = [
            &unsqueeze111_out1[..],
            &unsqueeze112_out1[..],
            &constant814_out1[..],
            &constant815_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape28_out1 = linear44_out1.reshape(concat55_out1);
        let transpose53_out1 = reshape28_out1.permute([0, 2, 1, 3]);
        let linear45_out1 = self.linear45.forward(mul89_out1.clone());
        let reshape29_out1 = linear45_out1.reshape(concat56_out1);
        let transpose54_out1 = reshape29_out1.permute([0, 2, 1, 3]);
        let linear46_out1 = self.linear46.forward(mul89_out1);
        let reshape30_out1 = linear46_out1.reshape(concat57_out1);
        let transpose55_out1 = reshape30_out1.permute([0, 2, 1, 3]);
        let mul90_out1 = transpose53_out1.clone().mul(unsqueeze33_out1.clone());
        let shape40_out1: [i64; 4] = {
            let axes = &transpose53_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant816_out1 = 3i64;
        let actual_idx = if constant816_out1 < 0 {
            (shape40_out1.len() as i64 + constant816_out1) as usize
        } else {
            constant816_out1 as usize
        };
        let gather56_out1 = shape40_out1[actual_idx] as i64;
        let constant817_out1 = 2i64;
        let div43_out1 = gather56_out1 / constant817_out1;
        let unsqueeze113_out1 = [div43_out1];
        let slice63_out1 = transpose53_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze113_out1[0]]);
        let unsqueeze114_out1 = [div43_out1];
        let slice64_out1 = transpose53_out1
            .slice(s![.., .., .., unsqueeze114_out1[0]..9223372036854775807]);
        let neg13_out1 = slice64_out1.neg();
        let concat58_out1 = burn::tensor::Tensor::cat(
            [neg13_out1, slice63_out1].into(),
            3,
        );
        let mul91_out1 = concat58_out1.mul(unsqueeze34_out1.clone());
        let add68_out1 = mul90_out1.add(mul91_out1);
        let mul92_out1 = transpose54_out1.clone().mul(unsqueeze33_out1.clone());
        let shape41_out1: [i64; 4] = {
            let axes = &transpose54_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant826_out1 = 3i64;
        let actual_idx = if constant826_out1 < 0 {
            (shape41_out1.len() as i64 + constant826_out1) as usize
        } else {
            constant826_out1 as usize
        };
        let gather57_out1 = shape41_out1[actual_idx] as i64;
        let constant827_out1 = 2i64;
        let div44_out1 = gather57_out1 / constant827_out1;
        let unsqueeze115_out1 = [div44_out1];
        let slice65_out1 = transpose54_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze115_out1[0]]);
        let unsqueeze116_out1 = [div44_out1];
        let slice66_out1 = transpose54_out1
            .slice(s![.., .., .., unsqueeze116_out1[0]..9223372036854775807]);
        let neg14_out1 = slice66_out1.neg();
        let concat59_out1 = burn::tensor::Tensor::cat(
            [neg14_out1, slice65_out1].into(),
            3,
        );
        let mul93_out1 = concat59_out1.mul(unsqueeze34_out1.clone());
        let add69_out1 = mul92_out1.add(mul93_out1);
        let concat60_out1 = burn::tensor::Tensor::cat(
            [past_key_6, add69_out1].into(),
            2,
        );
        let concat61_out1 = burn::tensor::Tensor::cat(
            [past_value_6, transpose55_out1].into(),
            2,
        );
        let slice67_out1 = concat60_out1.slice(s![.., .., - 72.., ..]);
        let slice68_out1 = concat61_out1.slice(s![.., .., - 72.., ..]);
        let shape42_out1: [i64; 4] = {
            let axes = &slice67_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant844_out1 = 2i64;
        let actual_idx = if constant844_out1 < 0 {
            (shape42_out1.len() as i64 + constant844_out1) as usize
        } else {
            constant844_out1 as usize
        };
        let gather58_out1 = shape42_out1[actual_idx] as i64;
        let unsqueeze117_out1 = [gather58_out1];
        let slice69_out1 = unsqueeze26_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze117_out1[0]]);
        let transpose56_out1 = slice67_out1.clone().permute([0, 1, 3, 2]);
        let constant849_out1 = 0.35355338f32;
        let mul94_out1 = add68_out1.mul_scalar(constant849_out1);
        let constant850_out1 = 0.35355338f32;
        let mul95_out1 = transpose56_out1.mul_scalar(constant850_out1);
        let matmul60_out1 = mul94_out1.matmul(mul95_out1);
        let add70_out1 = matmul60_out1.add(slice69_out1);
        let softmax7_out1 = burn::tensor::activation::softmax(add70_out1, 3);
        let matmul61_out1 = softmax7_out1.matmul(slice68_out1.clone());
        let transpose57_out1 = matmul61_out1.permute([0, 2, 1, 3]);
        let unsqueeze118_out1 = [gather54_out1];
        let unsqueeze119_out1 = [gather55_out1];
        let constant853_out1: [i64; 1] = [-1i64];
        let concat62_out1: [i64; 3usize] = [
            &unsqueeze118_out1[..],
            &unsqueeze119_out1[..],
            &constant853_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape31_out1 = transpose57_out1.reshape(concat62_out1);
        let linear47_out1 = self.linear47.forward(reshape31_out1);
        let mul96_out1 = constant27_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear47_out1);
        let add71_out1 = add66_out1.add(mul96_out1);
        let constant854_out1 = 2f32;
        let pow14_out1 = add71_out1.clone().powf_scalar(constant854_out1);
        let reducemean14_out1 = { pow14_out1.mean_dim(2usize) };
        let constant855_out1 = 0.00001f32;
        let add72_out1 = reducemean14_out1.add_scalar(constant855_out1);
        let sqrt14_out1 = add72_out1.sqrt();
        let constant856_out1 = 1f32;
        let div45_out1 = constant856_out1 / sqrt14_out1;
        let mul97_out1 = add71_out1.clone().mul(div45_out1);
        let mul98_out1 = constant26_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul97_out1);
        let linear48_out1 = self.linear48.forward(mul98_out1.clone());
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(linear48_out1.clone());
        let mul99_out1 = linear48_out1.mul(sigmoid7_out1);
        let linear49_out1 = self.linear49.forward(mul98_out1);
        let mul100_out1 = mul99_out1.mul(linear49_out1);
        let linear50_out1 = self.linear50.forward(mul100_out1);
        let mul101_out1 = constant28_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear50_out1);
        let add73_out1 = add71_out1.add(mul101_out1);
        let constant857_out1 = 2f32;
        let pow15_out1 = add73_out1.clone().powf_scalar(constant857_out1);
        let reducemean15_out1 = { pow15_out1.mean_dim(2usize) };
        let constant858_out1 = 0.00001f32;
        let add74_out1 = reducemean15_out1.add_scalar(constant858_out1);
        let sqrt15_out1 = add74_out1.sqrt();
        let constant859_out1 = 1f32;
        let div46_out1 = constant859_out1 / sqrt15_out1;
        let mul102_out1 = add73_out1.clone().mul(div46_out1);
        let mul103_out1 = constant29_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul102_out1);
        let shape43_out1: [i64; 3] = {
            let axes = &mul103_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant860_out1 = 0i64;
        let actual_idx = if constant860_out1 < 0 {
            (shape43_out1.len() as i64 + constant860_out1) as usize
        } else {
            constant860_out1 as usize
        };
        let gather59_out1 = shape43_out1[actual_idx] as i64;
        let constant861_out1 = 1i64;
        let actual_idx = if constant861_out1 < 0 {
            (shape43_out1.len() as i64 + constant861_out1) as usize
        } else {
            constant861_out1 as usize
        };
        let gather60_out1 = shape43_out1[actual_idx] as i64;
        let linear51_out1 = self.linear51.forward(mul103_out1.clone());
        let unsqueeze120_out1 = [gather59_out1];
        let unsqueeze121_out1 = [gather60_out1];
        let constant864_out1: [i64; 1] = [-1i64];
        let constant865_out1: [i64; 1] = [64i64];
        let concat63_out1: [i64; 4usize] = [
            &unsqueeze120_out1[..],
            &unsqueeze121_out1[..],
            &constant864_out1[..],
            &constant865_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze122_out1 = [gather59_out1];
        let unsqueeze123_out1 = [gather60_out1];
        let constant868_out1: [i64; 1] = [-1i64];
        let constant869_out1: [i64; 1] = [64i64];
        let concat64_out1: [i64; 4usize] = [
            &unsqueeze122_out1[..],
            &unsqueeze123_out1[..],
            &constant868_out1[..],
            &constant869_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let unsqueeze124_out1 = [gather59_out1];
        let unsqueeze125_out1 = [gather60_out1];
        let constant872_out1: [i64; 1] = [-1i64];
        let constant873_out1: [i64; 1] = [64i64];
        let concat65_out1: [i64; 4usize] = [
            &unsqueeze124_out1[..],
            &unsqueeze125_out1[..],
            &constant872_out1[..],
            &constant873_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape32_out1 = linear51_out1.reshape(concat63_out1);
        let transpose58_out1 = reshape32_out1.permute([0, 2, 1, 3]);
        let linear52_out1 = self.linear52.forward(mul103_out1.clone());
        let reshape33_out1 = linear52_out1.reshape(concat64_out1);
        let transpose59_out1 = reshape33_out1.permute([0, 2, 1, 3]);
        let linear53_out1 = self.linear53.forward(mul103_out1);
        let reshape34_out1 = linear53_out1.reshape(concat65_out1);
        let transpose60_out1 = reshape34_out1.permute([0, 2, 1, 3]);
        let mul104_out1 = transpose58_out1.clone().mul(unsqueeze33_out1.clone());
        let shape45_out1: [i64; 4] = {
            let axes = &transpose58_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant874_out1 = 3i64;
        let actual_idx = if constant874_out1 < 0 {
            (shape45_out1.len() as i64 + constant874_out1) as usize
        } else {
            constant874_out1 as usize
        };
        let gather61_out1 = shape45_out1[actual_idx] as i64;
        let constant875_out1 = 2i64;
        let div47_out1 = gather61_out1 / constant875_out1;
        let unsqueeze126_out1 = [div47_out1];
        let slice70_out1 = transpose58_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze126_out1[0]]);
        let unsqueeze127_out1 = [div47_out1];
        let slice71_out1 = transpose58_out1
            .slice(s![.., .., .., unsqueeze127_out1[0]..9223372036854775807]);
        let neg15_out1 = slice71_out1.neg();
        let concat66_out1 = burn::tensor::Tensor::cat(
            [neg15_out1, slice70_out1].into(),
            3,
        );
        let mul105_out1 = concat66_out1.mul(unsqueeze34_out1.clone());
        let add75_out1 = mul104_out1.add(mul105_out1);
        let mul106_out1 = transpose59_out1.clone().mul(unsqueeze33_out1);
        let shape46_out1: [i64; 4] = {
            let axes = &transpose59_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant884_out1 = 3i64;
        let actual_idx = if constant884_out1 < 0 {
            (shape46_out1.len() as i64 + constant884_out1) as usize
        } else {
            constant884_out1 as usize
        };
        let gather62_out1 = shape46_out1[actual_idx] as i64;
        let constant885_out1 = 2i64;
        let div48_out1 = gather62_out1 / constant885_out1;
        let unsqueeze128_out1 = [div48_out1];
        let slice72_out1 = transpose59_out1
            .clone()
            .slice(s![.., .., .., 0..unsqueeze128_out1[0]]);
        let unsqueeze129_out1 = [div48_out1];
        let slice73_out1 = transpose59_out1
            .slice(s![.., .., .., unsqueeze129_out1[0]..9223372036854775807]);
        let neg16_out1 = slice73_out1.neg();
        let concat67_out1 = burn::tensor::Tensor::cat(
            [neg16_out1, slice72_out1].into(),
            3,
        );
        let mul107_out1 = concat67_out1.mul(unsqueeze34_out1);
        let add76_out1 = mul106_out1.add(mul107_out1);
        let concat68_out1 = burn::tensor::Tensor::cat(
            [past_key_7, add76_out1].into(),
            2,
        );
        let concat69_out1 = burn::tensor::Tensor::cat(
            [past_value_7, transpose60_out1].into(),
            2,
        );
        let slice74_out1 = concat68_out1.slice(s![.., .., - 72.., ..]);
        let slice75_out1 = concat69_out1.slice(s![.., .., - 72.., ..]);
        let shape47_out1: [i64; 4] = {
            let axes = &slice74_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant902_out1 = 2i64;
        let actual_idx = if constant902_out1 < 0 {
            (shape47_out1.len() as i64 + constant902_out1) as usize
        } else {
            constant902_out1 as usize
        };
        let gather63_out1 = shape47_out1[actual_idx] as i64;
        let unsqueeze130_out1 = [gather63_out1];
        let slice76_out1 = unsqueeze26_out1
            .slice(s![.., .., .., 0..unsqueeze130_out1[0]]);
        let transpose61_out1 = slice74_out1.clone().permute([0, 1, 3, 2]);
        let constant907_out1 = 0.35355338f32;
        let mul108_out1 = add75_out1.mul_scalar(constant907_out1);
        let constant908_out1 = 0.35355338f32;
        let mul109_out1 = transpose61_out1.mul_scalar(constant908_out1);
        let matmul69_out1 = mul108_out1.matmul(mul109_out1);
        let add77_out1 = matmul69_out1.add(slice76_out1);
        let softmax8_out1 = burn::tensor::activation::softmax(add77_out1, 3);
        let matmul70_out1 = softmax8_out1.matmul(slice75_out1.clone());
        let transpose62_out1 = matmul70_out1.permute([0, 2, 1, 3]);
        let unsqueeze131_out1 = [gather59_out1];
        let unsqueeze132_out1 = [gather60_out1];
        let constant911_out1: [i64; 1] = [-1i64];
        let concat70_out1: [i64; 3usize] = [
            &unsqueeze131_out1[..],
            &unsqueeze132_out1[..],
            &constant911_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape35_out1 = transpose62_out1.reshape(concat70_out1);
        let linear54_out1 = self.linear54.forward(reshape35_out1);
        let mul110_out1 = constant31_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear54_out1);
        let add78_out1 = add73_out1.add(mul110_out1);
        let constant912_out1 = 2f32;
        let pow16_out1 = add78_out1.clone().powf_scalar(constant912_out1);
        let reducemean16_out1 = { pow16_out1.mean_dim(2usize) };
        let constant913_out1 = 0.00001f32;
        let add79_out1 = reducemean16_out1.add_scalar(constant913_out1);
        let sqrt16_out1 = add79_out1.sqrt();
        let constant914_out1 = 1f32;
        let div49_out1 = constant914_out1 / sqrt16_out1;
        let mul111_out1 = add78_out1.clone().mul(div49_out1);
        let mul112_out1 = constant30_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul111_out1);
        let linear55_out1 = self.linear55.forward(mul112_out1.clone());
        let sigmoid8_out1 = burn::tensor::activation::sigmoid(linear55_out1.clone());
        let mul113_out1 = linear55_out1.mul(sigmoid8_out1);
        let linear56_out1 = self.linear56.forward(mul112_out1);
        let mul114_out1 = mul113_out1.mul(linear56_out1);
        let linear57_out1 = self.linear57.forward(mul114_out1);
        let mul115_out1 = constant32_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear57_out1);
        let add80_out1 = add78_out1.add(mul115_out1);
        let constant915_out1 = 2f32;
        let pow17_out1 = add80_out1.clone().powf_scalar(constant915_out1);
        let reducemean17_out1 = { pow17_out1.mean_dim(2usize) };
        let constant916_out1 = 0.00001f32;
        let add81_out1 = reducemean17_out1.add_scalar(constant916_out1);
        let sqrt17_out1 = add81_out1.sqrt();
        let constant917_out1 = 1f32;
        let div50_out1 = constant917_out1 / sqrt17_out1;
        let mul116_out1 = add80_out1.mul(div50_out1);
        let mul117_out1 = constant33_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(mul116_out1);
        let linear58_out1 = self.linear58.forward(mul117_out1);
        let transpose63_out1 = linear58_out1.permute([0, 2, 1]);
        let concat71_out1 = burn::tensor::Tensor::cat(
            [latent_buffer, transpose63_out1].into(),
            2,
        );
        let shape48_out1: [i64; 3] = {
            let axes = &concat71_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant918_out1 = 2i64;
        let actual_idx = if constant918_out1 < 0 {
            (shape48_out1.len() as i64 + constant918_out1) as usize
        } else {
            constant918_out1 as usize
        };
        let gather64_out1 = shape48_out1[actual_idx] as i64;
        let constant921_out1 = 0.5f32;
        let greater2_out1 = is_last.greater_elem(constant921_out1);
        let constant922_out1 = self.constant922.val();
        let gather64_tensor_out1 = Tensor::<B, 1, _>::from_data([gather64_out1], &*self.device);
        let sub7_out1 = gather64_tensor_out1.clone().sub(constant922_out1);
        let clip17_out1 = sub7_out1.clamp_min(0i64);
        let where3_out1 = clip17_out1.mask_where(greater2_out1, gather64_tensor_out1);
        let gather65_out1 = {
            let indices = Tensor::<
                B,
                1,
                _,
            >::from_data([constant271_out1], &*self.device);
            let selected = Tensor::select(where3_out1.clone(), 0, indices);
            selected.into_scalar().elem::<i64>()
        };
        let slice77_out1 = concat71_out1.clone().slice(s![.., .., - 4..]);
        let concat72_out1 = burn::tensor::Tensor::cat(
            [conv_history.clone(), concat71_out1.clone()].into(),
            2,
        );
        let convtranspose1d1_out1 = self.convtranspose1d1.forward(concat72_out1);
        let shape49_out1: [i64; 3] = {
            let axes = &convtranspose1d1_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant929_out1 = 2i64;
        let actual_idx = if constant929_out1 < 0 {
            (shape49_out1.len() as i64 + constant929_out1) as usize
        } else {
            constant929_out1 as usize
        };
        let gather66_out1 = shape49_out1[actual_idx] as i64;
        let unsqueeze133_out1 = [gather66_out1];
        let slice78_out1 = convtranspose1d1_out1
            .slice(s![.., .., 0..unsqueeze133_out1[0]]);
        let pad2_out1 = slice78_out1
            .clone()
            .pad((6, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d4_out1 = self.conv1d4.forward(pad2_out1);
        let transpose65_out1 = conv1d4_out1.permute([0, 2, 1]);
        let reducemean18_out1 = { transpose65_out1.clone().mean_dim(2usize) };
        let sub13_out1 = transpose65_out1.sub(reducemean18_out1);
        let constant957_out1 = 2f32;
        let pow18_out1 = sub13_out1.clone().powf_scalar(constant957_out1);
        let reducemean19_out1 = { pow18_out1.mean_dim(2usize) };
        let constant958_out1 = 0.000001f32;
        let add87_out1 = reducemean19_out1.add_scalar(constant958_out1);
        let sqrt18_out1 = add87_out1.sqrt();
        let div52_out1 = sub13_out1.div(sqrt18_out1);
        let mul118_out1 = div52_out1
            .mul(constant77_out1.unsqueeze_dims(&[0isize, 1isize]));
        let add88_out1 = mul118_out1
            .add(constant78_out1.unsqueeze_dims(&[0isize, 1isize]));
        let linear59_out1 = self.linear59.forward(add88_out1);
        let constant959_out1 = 1.4142135f32;
        let div53_out1 = linear59_out1.clone().div_scalar(constant959_out1);
        let erf1_out1 = div53_out1.erf();
        let constant960_out1 = 1f32;
        let add89_out1 = erf1_out1.add_scalar(constant960_out1);
        let mul119_out1 = linear59_out1.mul(add89_out1);
        let constant961_out1 = 0.5f32;
        let mul120_out1 = mul119_out1.mul_scalar(constant961_out1);
        let linear60_out1 = self.linear60.forward(mul120_out1);
        let mul121_out1 = constant74_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear60_out1);
        let transpose66_out1 = mul121_out1.permute([0, 2, 1]);
        let add90_out1 = slice78_out1.add(transpose66_out1);
        let convtranspose1d2_out1 = self.convtranspose1d2.forward(add90_out1);
        let shape52_out1: [i64; 3] = {
            let axes = &convtranspose1d2_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant962_out1 = 2i64;
        let actual_idx = if constant962_out1 < 0 {
            (shape52_out1.len() as i64 + constant962_out1) as usize
        } else {
            constant962_out1 as usize
        };
        let gather69_out1 = shape52_out1[actual_idx] as i64;
        let unsqueeze136_out1 = [gather69_out1];
        let slice80_out1 = convtranspose1d2_out1
            .slice(s![.., .., 0..unsqueeze136_out1[0]]);
        let pad3_out1 = slice80_out1
            .clone()
            .pad((6, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d5_out1 = self.conv1d5.forward(pad3_out1);
        let transpose68_out1 = conv1d5_out1.permute([0, 2, 1]);
        let reducemean20_out1 = { transpose68_out1.clone().mean_dim(2usize) };
        let sub19_out1 = transpose68_out1.sub(reducemean20_out1);
        let constant990_out1 = 2f32;
        let pow19_out1 = sub19_out1.clone().powf_scalar(constant990_out1);
        let reducemean21_out1 = { pow19_out1.mean_dim(2usize) };
        let constant991_out1 = 0.000001f32;
        let add95_out1 = reducemean21_out1.add_scalar(constant991_out1);
        let sqrt19_out1 = add95_out1.sqrt();
        let div55_out1 = sub19_out1.div(sqrt19_out1);
        let mul122_out1 = div55_out1
            .mul(constant86_out1.unsqueeze_dims(&[0isize, 1isize]));
        let add96_out1 = mul122_out1
            .add(constant87_out1.unsqueeze_dims(&[0isize, 1isize]));
        let linear61_out1 = self.linear61.forward(add96_out1);
        let constant992_out1 = 1.4142135f32;
        let div56_out1 = linear61_out1.clone().div_scalar(constant992_out1);
        let erf2_out1 = div56_out1.erf();
        let constant993_out1 = 1f32;
        let add97_out1 = erf2_out1.add_scalar(constant993_out1);
        let mul123_out1 = linear61_out1.mul(add97_out1);
        let constant994_out1 = 0.5f32;
        let mul124_out1 = mul123_out1.mul_scalar(constant994_out1);
        let linear62_out1 = self.linear62.forward(mul124_out1);
        let mul125_out1 = constant83_out1
            .unsqueeze_dims(&[0isize, 1isize])
            .mul(linear62_out1);
        let transpose69_out1 = mul125_out1.permute([0, 2, 1]);
        let add98_out1 = slice80_out1.add(transpose69_out1);
        let pad4_out1 = add98_out1
            .pad((6, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d6_out1 = self.conv1d6.forward(pad4_out1);
        let exp1_out1 = constant212_out1.exp();
        let exp2_out1 = constant213_out1.exp();
        let constant1018_out1 = 0.000000001f32;
        let add103_out1 = exp2_out1.add_scalar(constant1018_out1);
        let reciprocal1_out1 = add103_out1.recip();
        let mul127_out1 = conv1d6_out1.clone().mul(exp1_out1);
        let sin2_out1 = mul127_out1.sin();
        let constant1020_out1 = 2f32;
        let pow20_out1 = sin2_out1.powf_scalar(constant1020_out1);
        let mul128_out1 = reciprocal1_out1.mul(pow20_out1);
        let add104_out1 = conv1d6_out1.add(mul128_out1);
        let convtranspose1d3_out1 = self.convtranspose1d3.forward(add104_out1);
        let shape57_out1: [i64; 3] = {
            let axes = &convtranspose1d3_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant1021_out1 = 2i64;
        let actual_idx = if constant1021_out1 < 0 {
            (shape57_out1.len() as i64 + constant1021_out1) as usize
        } else {
            constant1021_out1 as usize
        };
        let gather74_out1 = shape57_out1[actual_idx] as i64;
        let constant1022_out1 = 8i64;
        let sub25_out1 = gather74_out1 - constant1022_out1;
        let unsqueeze141_out1 = [sub25_out1];
        let slice83_out1 = convtranspose1d3_out1
            .slice(s![.., .., 8..unsqueeze141_out1[0]]);
        let exp3_out1 = constant214_out1.exp();
        let exp4_out1 = constant215_out1.exp();
        let constant1027_out1 = 0.000000001f32;
        let add105_out1 = exp4_out1.add_scalar(constant1027_out1);
        let reciprocal2_out1 = add105_out1.recip();
        let mul130_out1 = slice83_out1.clone().mul(exp3_out1);
        let sin3_out1 = mul130_out1.sin();
        let constant1029_out1 = 2f32;
        let pow21_out1 = sin3_out1.powf_scalar(constant1029_out1);
        let mul131_out1 = reciprocal2_out1.mul(pow21_out1);
        let add106_out1 = slice83_out1.clone().add(mul131_out1);
        let pad5_out1 = add106_out1
            .pad((6, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d7_out1 = self.conv1d7.forward(pad5_out1);
        let exp5_out1 = constant216_out1.exp();
        let exp6_out1 = constant217_out1.exp();
        let constant1053_out1 = 0.000000001f32;
        let add111_out1 = exp6_out1.add_scalar(constant1053_out1);
        let reciprocal3_out1 = add111_out1.recip();
        let mul133_out1 = conv1d7_out1.clone().mul(exp5_out1);
        let sin4_out1 = mul133_out1.sin();
        let constant1055_out1 = 2f32;
        let pow22_out1 = sin4_out1.powf_scalar(constant1055_out1);
        let mul134_out1 = reciprocal3_out1.mul(pow22_out1);
        let add112_out1 = conv1d7_out1.add(mul134_out1);
        let conv1d8_out1 = self.conv1d8.forward(add112_out1);
        let add116_out1 = conv1d8_out1.add(slice83_out1);
        let exp7_out1 = constant218_out1.exp();
        let exp8_out1 = constant219_out1.exp();
        let constant1078_out1 = 0.000000001f32;
        let add117_out1 = exp8_out1.add_scalar(constant1078_out1);
        let reciprocal4_out1 = add117_out1.recip();
        let mul136_out1 = add116_out1.clone().mul(exp7_out1);
        let sin5_out1 = mul136_out1.sin();
        let constant1080_out1 = 2f32;
        let pow23_out1 = sin5_out1.powf_scalar(constant1080_out1);
        let mul137_out1 = reciprocal4_out1.mul(pow23_out1);
        let add118_out1 = add116_out1.clone().add(mul137_out1);
        let pad7_out1 = add118_out1
            .pad((18, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d9_out1 = self.conv1d9.forward(pad7_out1);
        let exp9_out1 = constant220_out1.exp();
        let exp10_out1 = constant221_out1.exp();
        let constant1104_out1 = 0.000000001f32;
        let add123_out1 = exp10_out1.add_scalar(constant1104_out1);
        let reciprocal5_out1 = add123_out1.recip();
        let mul139_out1 = conv1d9_out1.clone().mul(exp9_out1);
        let sin6_out1 = mul139_out1.sin();
        let constant1106_out1 = 2f32;
        let pow24_out1 = sin6_out1.powf_scalar(constant1106_out1);
        let mul140_out1 = reciprocal5_out1.mul(pow24_out1);
        let add124_out1 = conv1d9_out1.add(mul140_out1);
        let conv1d10_out1 = self.conv1d10.forward(add124_out1);
        let add128_out1 = conv1d10_out1.add(add116_out1);
        let exp11_out1 = constant222_out1.exp();
        let exp12_out1 = constant223_out1.exp();
        let constant1129_out1 = 0.000000001f32;
        let add129_out1 = exp12_out1.add_scalar(constant1129_out1);
        let reciprocal6_out1 = add129_out1.recip();
        let mul142_out1 = add128_out1.clone().mul(exp11_out1);
        let sin7_out1 = mul142_out1.sin();
        let constant1131_out1 = 2f32;
        let pow25_out1 = sin7_out1.powf_scalar(constant1131_out1);
        let mul143_out1 = reciprocal6_out1.mul(pow25_out1);
        let add130_out1 = add128_out1.clone().add(mul143_out1);
        let pad9_out1 = add130_out1
            .pad((54, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d11_out1 = self.conv1d11.forward(pad9_out1);
        let exp13_out1 = constant224_out1.exp();
        let exp14_out1 = constant225_out1.exp();
        let constant1155_out1 = 0.000000001f32;
        let add135_out1 = exp14_out1.add_scalar(constant1155_out1);
        let reciprocal7_out1 = add135_out1.recip();
        let mul145_out1 = conv1d11_out1.clone().mul(exp13_out1);
        let sin8_out1 = mul145_out1.sin();
        let constant1157_out1 = 2f32;
        let pow26_out1 = sin8_out1.powf_scalar(constant1157_out1);
        let mul146_out1 = reciprocal7_out1.mul(pow26_out1);
        let add136_out1 = conv1d11_out1.add(mul146_out1);
        let conv1d12_out1 = self.conv1d12.forward(add136_out1);
        let add140_out1 = conv1d12_out1.add(add128_out1);
        let exp15_out1 = constant226_out1.exp();
        let exp16_out1 = constant227_out1.exp();
        let constant1180_out1 = 0.000000001f32;
        let add141_out1 = exp16_out1.add_scalar(constant1180_out1);
        let reciprocal8_out1 = add141_out1.recip();
        let mul148_out1 = add140_out1.clone().mul(exp15_out1);
        let sin9_out1 = mul148_out1.sin();
        let constant1182_out1 = 2f32;
        let pow27_out1 = sin9_out1.powf_scalar(constant1182_out1);
        let mul149_out1 = reciprocal8_out1.mul(pow27_out1);
        let add142_out1 = add140_out1.add(mul149_out1);
        let convtranspose1d4_out1 = self.convtranspose1d4.forward(add142_out1);
        let shape70_out1: [i64; 3] = {
            let axes = &convtranspose1d4_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant1183_out1 = 2i64;
        let actual_idx = if constant1183_out1 < 0 {
            (shape70_out1.len() as i64 + constant1183_out1) as usize
        } else {
            constant1183_out1 as usize
        };
        let gather87_out1 = shape70_out1[actual_idx] as i64;
        let constant1184_out1 = 5i64;
        let sub56_out1 = gather87_out1 - constant1184_out1;
        let unsqueeze154_out1 = [sub56_out1];
        let slice90_out1 = convtranspose1d4_out1
            .slice(s![.., .., 5..unsqueeze154_out1[0]]);
        let exp17_out1 = constant228_out1.exp();
        let exp18_out1 = constant229_out1.exp();
        let constant1189_out1 = 0.000000001f32;
        let add143_out1 = exp18_out1.add_scalar(constant1189_out1);
        let reciprocal9_out1 = add143_out1.recip();
        let mul151_out1 = slice90_out1.clone().mul(exp17_out1);
        let sin10_out1 = mul151_out1.sin();
        let constant1191_out1 = 2f32;
        let pow28_out1 = sin10_out1.powf_scalar(constant1191_out1);
        let mul152_out1 = reciprocal9_out1.mul(pow28_out1);
        let add144_out1 = slice90_out1.clone().add(mul152_out1);
        let pad11_out1 = add144_out1
            .pad((6, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d13_out1 = self.conv1d13.forward(pad11_out1);
        let exp19_out1 = constant230_out1.exp();
        let exp20_out1 = constant231_out1.exp();
        let constant1215_out1 = 0.000000001f32;
        let add149_out1 = exp20_out1.add_scalar(constant1215_out1);
        let reciprocal10_out1 = add149_out1.recip();
        let mul154_out1 = conv1d13_out1.clone().mul(exp19_out1);
        let sin11_out1 = mul154_out1.sin();
        let constant1217_out1 = 2f32;
        let pow29_out1 = sin11_out1.powf_scalar(constant1217_out1);
        let mul155_out1 = reciprocal10_out1.mul(pow29_out1);
        let add150_out1 = conv1d13_out1.add(mul155_out1);
        let conv1d14_out1 = self.conv1d14.forward(add150_out1);
        let add154_out1 = conv1d14_out1.add(slice90_out1);
        let exp21_out1 = constant232_out1.exp();
        let exp22_out1 = constant233_out1.exp();
        let constant1240_out1 = 0.000000001f32;
        let add155_out1 = exp22_out1.add_scalar(constant1240_out1);
        let reciprocal11_out1 = add155_out1.recip();
        let mul157_out1 = add154_out1.clone().mul(exp21_out1);
        let sin12_out1 = mul157_out1.sin();
        let constant1242_out1 = 2f32;
        let pow30_out1 = sin12_out1.powf_scalar(constant1242_out1);
        let mul158_out1 = reciprocal11_out1.mul(pow30_out1);
        let add156_out1 = add154_out1.clone().add(mul158_out1);
        let pad13_out1 = add156_out1
            .pad((18, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d15_out1 = self.conv1d15.forward(pad13_out1);
        let exp23_out1 = constant234_out1.exp();
        let exp24_out1 = constant235_out1.exp();
        let constant1266_out1 = 0.000000001f32;
        let add161_out1 = exp24_out1.add_scalar(constant1266_out1);
        let reciprocal12_out1 = add161_out1.recip();
        let mul160_out1 = conv1d15_out1.clone().mul(exp23_out1);
        let sin13_out1 = mul160_out1.sin();
        let constant1268_out1 = 2f32;
        let pow31_out1 = sin13_out1.powf_scalar(constant1268_out1);
        let mul161_out1 = reciprocal12_out1.mul(pow31_out1);
        let add162_out1 = conv1d15_out1.add(mul161_out1);
        let conv1d16_out1 = self.conv1d16.forward(add162_out1);
        let add166_out1 = conv1d16_out1.add(add154_out1);
        let exp25_out1 = constant236_out1.exp();
        let exp26_out1 = constant237_out1.exp();
        let constant1291_out1 = 0.000000001f32;
        let add167_out1 = exp26_out1.add_scalar(constant1291_out1);
        let reciprocal13_out1 = add167_out1.recip();
        let mul163_out1 = add166_out1.clone().mul(exp25_out1);
        let sin14_out1 = mul163_out1.sin();
        let constant1293_out1 = 2f32;
        let pow32_out1 = sin14_out1.powf_scalar(constant1293_out1);
        let mul164_out1 = reciprocal13_out1.mul(pow32_out1);
        let add168_out1 = add166_out1.clone().add(mul164_out1);
        let pad15_out1 = add168_out1
            .pad((54, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d17_out1 = self.conv1d17.forward(pad15_out1);
        let exp27_out1 = constant238_out1.exp();
        let exp28_out1 = constant239_out1.exp();
        let constant1317_out1 = 0.000000001f32;
        let add173_out1 = exp28_out1.add_scalar(constant1317_out1);
        let reciprocal14_out1 = add173_out1.recip();
        let mul166_out1 = conv1d17_out1.clone().mul(exp27_out1);
        let sin15_out1 = mul166_out1.sin();
        let constant1319_out1 = 2f32;
        let pow33_out1 = sin15_out1.powf_scalar(constant1319_out1);
        let mul167_out1 = reciprocal14_out1.mul(pow33_out1);
        let add174_out1 = conv1d17_out1.add(mul167_out1);
        let conv1d18_out1 = self.conv1d18.forward(add174_out1);
        let add178_out1 = conv1d18_out1.add(add166_out1);
        let exp29_out1 = constant240_out1.exp();
        let exp30_out1 = constant241_out1.exp();
        let constant1342_out1 = 0.000000001f32;
        let add179_out1 = exp30_out1.add_scalar(constant1342_out1);
        let reciprocal15_out1 = add179_out1.recip();
        let mul169_out1 = add178_out1.clone().mul(exp29_out1);
        let sin16_out1 = mul169_out1.sin();
        let constant1344_out1 = 2f32;
        let pow34_out1 = sin16_out1.powf_scalar(constant1344_out1);
        let mul170_out1 = reciprocal15_out1.mul(pow34_out1);
        let add180_out1 = add178_out1.add(mul170_out1);
        let convtranspose1d5_out1 = self.convtranspose1d5.forward(add180_out1);
        let shape83_out1: [i64; 3] = {
            let axes = &convtranspose1d5_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant1345_out1 = 2i64;
        let actual_idx = if constant1345_out1 < 0 {
            (shape83_out1.len() as i64 + constant1345_out1) as usize
        } else {
            constant1345_out1 as usize
        };
        let gather100_out1 = shape83_out1[actual_idx] as i64;
        let constant1346_out1 = 4i64;
        let sub87_out1 = gather100_out1 - constant1346_out1;
        let unsqueeze167_out1 = [sub87_out1];
        let slice97_out1 = convtranspose1d5_out1
            .slice(s![.., .., 4..unsqueeze167_out1[0]]);
        let exp31_out1 = constant242_out1.exp();
        let exp32_out1 = constant243_out1.exp();
        let constant1351_out1 = 0.000000001f32;
        let add181_out1 = exp32_out1.add_scalar(constant1351_out1);
        let reciprocal16_out1 = add181_out1.recip();
        let mul172_out1 = slice97_out1.clone().mul(exp31_out1);
        let sin17_out1 = mul172_out1.sin();
        let constant1353_out1 = 2f32;
        let pow35_out1 = sin17_out1.powf_scalar(constant1353_out1);
        let mul173_out1 = reciprocal16_out1.mul(pow35_out1);
        let add182_out1 = slice97_out1.clone().add(mul173_out1);
        let pad17_out1 = add182_out1
            .pad((6, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d19_out1 = self.conv1d19.forward(pad17_out1);
        let exp33_out1 = constant244_out1.exp();
        let exp34_out1 = constant245_out1.exp();
        let constant1377_out1 = 0.000000001f32;
        let add187_out1 = exp34_out1.add_scalar(constant1377_out1);
        let reciprocal17_out1 = add187_out1.recip();
        let mul175_out1 = conv1d19_out1.clone().mul(exp33_out1);
        let sin18_out1 = mul175_out1.sin();
        let constant1379_out1 = 2f32;
        let pow36_out1 = sin18_out1.powf_scalar(constant1379_out1);
        let mul176_out1 = reciprocal17_out1.mul(pow36_out1);
        let add188_out1 = conv1d19_out1.add(mul176_out1);
        let conv1d20_out1 = self.conv1d20.forward(add188_out1);
        let add192_out1 = conv1d20_out1.add(slice97_out1);
        let exp35_out1 = constant246_out1.exp();
        let exp36_out1 = constant247_out1.exp();
        let constant1402_out1 = 0.000000001f32;
        let add193_out1 = exp36_out1.add_scalar(constant1402_out1);
        let reciprocal18_out1 = add193_out1.recip();
        let mul178_out1 = add192_out1.clone().mul(exp35_out1);
        let sin19_out1 = mul178_out1.sin();
        let constant1404_out1 = 2f32;
        let pow37_out1 = sin19_out1.powf_scalar(constant1404_out1);
        let mul179_out1 = reciprocal18_out1.mul(pow37_out1);
        let add194_out1 = add192_out1.clone().add(mul179_out1);
        let pad19_out1 = add194_out1
            .pad((18, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d21_out1 = self.conv1d21.forward(pad19_out1);
        let exp37_out1 = constant248_out1.exp();
        let exp38_out1 = constant249_out1.exp();
        let constant1428_out1 = 0.000000001f32;
        let add199_out1 = exp38_out1.add_scalar(constant1428_out1);
        let reciprocal19_out1 = add199_out1.recip();
        let mul181_out1 = conv1d21_out1.clone().mul(exp37_out1);
        let sin20_out1 = mul181_out1.sin();
        let constant1430_out1 = 2f32;
        let pow38_out1 = sin20_out1.powf_scalar(constant1430_out1);
        let mul182_out1 = reciprocal19_out1.mul(pow38_out1);
        let add200_out1 = conv1d21_out1.add(mul182_out1);
        let conv1d22_out1 = self.conv1d22.forward(add200_out1);
        let add204_out1 = conv1d22_out1.add(add192_out1);
        let exp39_out1 = constant250_out1.exp();
        let exp40_out1 = constant251_out1.exp();
        let constant1453_out1 = 0.000000001f32;
        let add205_out1 = exp40_out1.add_scalar(constant1453_out1);
        let reciprocal20_out1 = add205_out1.recip();
        let mul184_out1 = add204_out1.clone().mul(exp39_out1);
        let sin21_out1 = mul184_out1.sin();
        let constant1455_out1 = 2f32;
        let pow39_out1 = sin21_out1.powf_scalar(constant1455_out1);
        let mul185_out1 = reciprocal20_out1.mul(pow39_out1);
        let add206_out1 = add204_out1.clone().add(mul185_out1);
        let pad21_out1 = add206_out1
            .pad((54, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d23_out1 = self.conv1d23.forward(pad21_out1);
        let exp41_out1 = constant252_out1.exp();
        let exp42_out1 = constant253_out1.exp();
        let constant1479_out1 = 0.000000001f32;
        let add211_out1 = exp42_out1.add_scalar(constant1479_out1);
        let reciprocal21_out1 = add211_out1.recip();
        let mul187_out1 = conv1d23_out1.clone().mul(exp41_out1);
        let sin22_out1 = mul187_out1.sin();
        let constant1481_out1 = 2f32;
        let pow40_out1 = sin22_out1.powf_scalar(constant1481_out1);
        let mul188_out1 = reciprocal21_out1.mul(pow40_out1);
        let add212_out1 = conv1d23_out1.add(mul188_out1);
        let conv1d24_out1 = self.conv1d24.forward(add212_out1);
        let add216_out1 = conv1d24_out1.add(add204_out1);
        let exp43_out1 = constant254_out1.exp();
        let exp44_out1 = constant255_out1.exp();
        let constant1504_out1 = 0.000000001f32;
        let add217_out1 = exp44_out1.add_scalar(constant1504_out1);
        let reciprocal22_out1 = add217_out1.recip();
        let mul190_out1 = add216_out1.clone().mul(exp43_out1);
        let sin23_out1 = mul190_out1.sin();
        let constant1506_out1 = 2f32;
        let pow41_out1 = sin23_out1.powf_scalar(constant1506_out1);
        let mul191_out1 = reciprocal22_out1.mul(pow41_out1);
        let add218_out1 = add216_out1.add(mul191_out1);
        let convtranspose1d6_out1 = self.convtranspose1d6.forward(add218_out1);
        let shape96_out1: [i64; 3] = {
            let axes = &convtranspose1d6_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant1507_out1 = 2i64;
        let actual_idx = if constant1507_out1 < 0 {
            (shape96_out1.len() as i64 + constant1507_out1) as usize
        } else {
            constant1507_out1 as usize
        };
        let gather113_out1 = shape96_out1[actual_idx] as i64;
        let constant1508_out1 = 3i64;
        let sub118_out1 = gather113_out1 - constant1508_out1;
        let unsqueeze180_out1 = [sub118_out1];
        let slice104_out1 = convtranspose1d6_out1
            .slice(s![.., .., 3..unsqueeze180_out1[0]]);
        let exp45_out1 = constant256_out1.exp();
        let exp46_out1 = constant257_out1.exp();
        let constant1513_out1 = 0.000000001f32;
        let add219_out1 = exp46_out1.add_scalar(constant1513_out1);
        let reciprocal23_out1 = add219_out1.recip();
        let mul193_out1 = slice104_out1.clone().mul(exp45_out1);
        let sin24_out1 = mul193_out1.sin();
        let constant1515_out1 = 2f32;
        let pow42_out1 = sin24_out1.powf_scalar(constant1515_out1);
        let mul194_out1 = reciprocal23_out1.mul(pow42_out1);
        let add220_out1 = slice104_out1.clone().add(mul194_out1);
        let pad23_out1 = add220_out1
            .pad((6, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d25_out1 = self.conv1d25.forward(pad23_out1);
        let exp47_out1 = constant258_out1.exp();
        let exp48_out1 = constant259_out1.exp();
        let constant1539_out1 = 0.000000001f32;
        let add225_out1 = exp48_out1.add_scalar(constant1539_out1);
        let reciprocal24_out1 = add225_out1.recip();
        let mul196_out1 = conv1d25_out1.clone().mul(exp47_out1);
        let sin25_out1 = mul196_out1.sin();
        let constant1541_out1 = 2f32;
        let pow43_out1 = sin25_out1.powf_scalar(constant1541_out1);
        let mul197_out1 = reciprocal24_out1.mul(pow43_out1);
        let add226_out1 = conv1d25_out1.add(mul197_out1);
        let conv1d26_out1 = self.conv1d26.forward(add226_out1);
        let add230_out1 = conv1d26_out1.add(slice104_out1);
        let exp49_out1 = constant260_out1.exp();
        let exp50_out1 = constant261_out1.exp();
        let constant1564_out1 = 0.000000001f32;
        let add231_out1 = exp50_out1.add_scalar(constant1564_out1);
        let reciprocal25_out1 = add231_out1.recip();
        let mul199_out1 = add230_out1.clone().mul(exp49_out1);
        let sin26_out1 = mul199_out1.sin();
        let constant1566_out1 = 2f32;
        let pow44_out1 = sin26_out1.powf_scalar(constant1566_out1);
        let mul200_out1 = reciprocal25_out1.mul(pow44_out1);
        let add232_out1 = add230_out1.clone().add(mul200_out1);
        let pad25_out1 = add232_out1
            .pad((18, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d27_out1 = self.conv1d27.forward(pad25_out1);
        let exp51_out1 = constant262_out1.exp();
        let exp52_out1 = constant263_out1.exp();
        let constant1590_out1 = 0.000000001f32;
        let add237_out1 = exp52_out1.add_scalar(constant1590_out1);
        let reciprocal26_out1 = add237_out1.recip();
        let mul202_out1 = conv1d27_out1.clone().mul(exp51_out1);
        let sin27_out1 = mul202_out1.sin();
        let constant1592_out1 = 2f32;
        let pow45_out1 = sin27_out1.powf_scalar(constant1592_out1);
        let mul203_out1 = reciprocal26_out1.mul(pow45_out1);
        let add238_out1 = conv1d27_out1.add(mul203_out1);
        let conv1d28_out1 = self.conv1d28.forward(add238_out1);
        let add242_out1 = conv1d28_out1.add(add230_out1);
        let exp53_out1 = constant264_out1.exp();
        let exp54_out1 = constant265_out1.exp();
        let constant1615_out1 = 0.000000001f32;
        let add243_out1 = exp54_out1.add_scalar(constant1615_out1);
        let reciprocal27_out1 = add243_out1.recip();
        let mul205_out1 = add242_out1.clone().mul(exp53_out1);
        let sin28_out1 = mul205_out1.sin();
        let constant1617_out1 = 2f32;
        let pow46_out1 = sin28_out1.powf_scalar(constant1617_out1);
        let mul206_out1 = reciprocal27_out1.mul(pow46_out1);
        let add244_out1 = add242_out1.clone().add(mul206_out1);
        let pad27_out1 = add244_out1
            .pad((54, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d29_out1 = self.conv1d29.forward(pad27_out1);
        let exp55_out1 = constant266_out1.exp();
        let exp56_out1 = constant267_out1.exp();
        let constant1641_out1 = 0.000000001f32;
        let add249_out1 = exp56_out1.add_scalar(constant1641_out1);
        let reciprocal28_out1 = add249_out1.recip();
        let mul208_out1 = conv1d29_out1.clone().mul(exp55_out1);
        let sin29_out1 = mul208_out1.sin();
        let constant1643_out1 = 2f32;
        let pow47_out1 = sin29_out1.powf_scalar(constant1643_out1);
        let mul209_out1 = reciprocal28_out1.mul(pow47_out1);
        let add250_out1 = conv1d29_out1.add(mul209_out1);
        let conv1d30_out1 = self.conv1d30.forward(add250_out1);
        let add254_out1 = conv1d30_out1.add(add242_out1);
        let exp57_out1 = constant268_out1.exp();
        let exp58_out1 = constant269_out1.exp();
        let constant1666_out1 = 0.000000001f32;
        let add255_out1 = exp58_out1.add_scalar(constant1666_out1);
        let reciprocal29_out1 = add255_out1.recip();
        let mul211_out1 = add254_out1.clone().mul(exp57_out1);
        let sin30_out1 = mul211_out1.sin();
        let constant1668_out1 = 2f32;
        let pow48_out1 = sin30_out1.powf_scalar(constant1668_out1);
        let mul212_out1 = reciprocal29_out1.mul(pow48_out1);
        let add256_out1 = add254_out1.add(mul212_out1);
        let pad29_out1 = add256_out1
            .pad((6, 0, 0, 0), burn::tensor::ops::PadMode::Constant(0_f32));
        let conv1d31_out1 = self.conv1d31.forward(pad29_out1);
        let squeeze17_out1 = conv1d31_out1.squeeze_dims::<2>(&[1]);
        let clip18_out1 = squeeze17_out1.clamp(-1f64, 1f64);
        let shape111_out1: [i64; 3] = {
            let axes = &conv_history.dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let constant1695_out1 = 2i64;
        let actual_idx = if constant1695_out1 < 0 {
            (shape111_out1.len() as i64 + constant1695_out1) as usize
        } else {
            constant1695_out1 as usize
        };
        let gather128_out1 = shape111_out1[actual_idx] as i64;
        let constant1697_out1 = self.constant1697.val();
        let gather128_tensor_out1 = Tensor::<B, 1, _>::from_data([gather128_out1], &*self.device);
        let mul213_out1 = gather128_tensor_out1.mul(constant1697_out1);
        let shape112_out1: [i64; 2] = {
            let axes = &clip18_out1.clone().dims()[0..2];
            let mut output = [0i64; 2];
            for i in 0..2 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice112_out1: [i64; 1] = shape112_out1[1..2].try_into().unwrap();
        let squeeze18_out1 = slice112_out1[0] as i64;
        let constant1703_out1 = self.constant1703.val();
        let mul214_out1 = where3_out1.mul(constant1703_out1);
        let add263_out1 = mul213_out1.clone().add(mul214_out1);
        let squeeze18_tensor_out1 = Tensor::<B, 1, _>::from_data([squeeze18_out1], &*self.device);
        let min1_out1 = squeeze18_tensor_out1.min_pair(add263_out1);
        let sub154_out1 = min1_out1.sub(mul213_out1.clone());
        let gather129_out1 = {
            let indices = Tensor::<
                B,
                1,
                _,
            >::from_data([constant271_out1], &*self.device);
            let selected = Tensor::select(mul213_out1.clone(), 0, indices);
            selected.into_scalar().elem::<i64>()
        };
        let unsqueeze195_out1 = [gather129_out1];
        let slice113_out1 = clip18_out1
            .slice(s![.., unsqueeze195_out1[0]..9223372036854775807]);
        let unsqueeze196_out1 = [gather65_out1];
        let slice114_out1 = concat71_out1.slice(s![.., .., 0..unsqueeze196_out1[0]]);
        let slice115_out1 = slice114_out1.slice(s![.., .., - 4..]);
        (
            slice113_out1,
            sub154_out1,
            slice20_out1,
            slice77_out1,
            slice115_out1,
            slice25_out1,
            slice32_out1,
            slice39_out1,
            slice46_out1,
            slice53_out1,
            slice60_out1,
            slice67_out1,
            slice74_out1,
            slice26_out1,
            slice33_out1,
            slice40_out1,
            slice47_out1,
            slice54_out1,
            slice61_out1,
            slice68_out1,
            slice75_out1,
        )
    }
}
