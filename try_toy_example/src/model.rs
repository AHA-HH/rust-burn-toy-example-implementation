use burn::prelude::Module;              // Module derive/trait
use burn::config::Config;               // Config derive/trait, for reproducibility
use burn::nn::{Linear, LinearConfig};   // layers
use burn::nn::Tanh;                     // activation
use burn::tensor::backend::Backend;     // Backend trait
use burn::tensor::Tensor;               // Tensor type

// Two-Layer Network Module Definition

#[derive(Module, Debug)]
pub struct TwoLayerNet<B: Backend> {
    linear1: Linear<B>,
    activation: Tanh,
    linear2: Linear<B>,
}

#[derive(Config, Debug)]
pub struct TwoLayerNetConfig {
    pub input_features: usize,
    pub hidden_features: usize,
    pub output_features: usize,
}

impl TwoLayerNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TwoLayerNet<B> {
        TwoLayerNet {
            linear1: LinearConfig::new(self.input_features, self.hidden_features).init(device),
            activation: Tanh::new(),
            linear2: LinearConfig::new(self.hidden_features, self.output_features).init(device),
        }
    }
}

// impl<B: Backend> TwoLayerNet<B> {
//     pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        
//         let x = self.linear1.forward(input);
//         let x = self.activation.forward(x);
//         let x = self.linear2.forward(x);
//         x
//     }
// }

// Override with Padding, change to n - 2
impl<B: Backend> TwoLayerNet<B> {
    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let n = input.dims()[0];
        let input_reshaped = input.clone().reshape([n, 1]);
        let x = self.linear1.forward(input_reshaped);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
	    let raw_output = x.reshape([n]);

        let enforced_output = (Tensor::<B, 1>::ones([n], &raw_output.device()) 
        - input.clone().powf_scalar(2.0)) * raw_output;

        enforced_output
    }
}