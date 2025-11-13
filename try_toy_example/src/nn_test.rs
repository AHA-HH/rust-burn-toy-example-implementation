// Simple two layer neural network to understand how Burn framework works
use burn::{
    backend::NdArray,
    config::Config,
    module::Module,
    nn::{self, LinearConfig, Tanh},
    optim::{GradientsParams, Optimizer, SgdConfig},
    tensor::{Tensor, backend::Backend},
};

use burn_autodiff::Autodiff;

// Define our Autodiff backend type
type MyAutodiffBackend = Autodiff<NdArray>;

// Two-Layer Network Module Definition

#[derive(Module, Debug)]
pub struct TwoLayerNet<B: Backend> {
    linear1: nn::Linear<B>,
    activation: Tanh,
    linear2: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct TwoLayerNetConfig {
    input_features: usize,
    hidden_features: usize,
    output_features: usize,
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

impl<B: Backend> TwoLayerNet<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        x
    }
}

// Training Function for Two-Layer Network

pub fn train_two_layer_net() {
    let device = Default::default();
    let learning_rate = 0.01;
    let hidden_size = 20;

    let n_points = 100;
    let x_vals: Vec<f32> = (0..n_points)
        .map(|i| i as f32 / (n_points as f32 - 1.0)) // 0.0 → 1.0
        .collect();

    let y_vals: Vec<f32> = x_vals.iter().map(|&x| (2.0 * std::f32::consts::PI * x).sin()).collect();

    // reshape into [n_points, 1]
    let input_x = Tensor::<MyAutodiffBackend, 1>::from_floats(x_vals.as_slice(), &device)
        .reshape([x_vals.len(), 1]);

    let target_y = Tensor::<MyAutodiffBackend, 1>::from_floats(y_vals.as_slice(), &device)
        .reshape([y_vals.len(), 1]);

    let config = TwoLayerNetConfig::new(1, hidden_size, 1);
    let mut model = config.init(&device);

    let optimizer_config = SgdConfig::new();
    let mut optimizer = optimizer_config.init();

    println!("\nTraining the Two-Layer Network");

    for i in 0..5000 {
        let output_y = model.forward(input_x.clone());
        let loss = (output_y.clone() - target_y.clone())
            .powf_scalar(2.0)
            .mean();
        let gradients = loss.backward();

        model = optimizer.step(
            learning_rate.into(),
            model.clone(),
            GradientsParams::from_grads(gradients, &model),
        );

        println!("Step {}: Loss: {:.4}", i + 1, loss.to_data());
    }

    println!("Training Finished");
    let test_x: Vec<f32> = (0..100)
        .map(|i| i as f32 / 99.0)
        .collect();

    let test_input = Tensor::<MyAutodiffBackend, 1>::from_floats(test_x.as_slice(), &device)
        .reshape([100, 1]);

    let pred_y = model.forward(test_input.clone());

    println!("x\ty_true\ty_pred");
    let pred_data = pred_y.to_data().to_vec::<f32>().unwrap();
    for (x, y_pred) in test_x.iter().zip(pred_data.iter()) {
        println!("{:.3}\t{:.3}\t{:.3}", x, (2.0 * std::f32::consts::PI * x).sin(), y_pred);
    }
    println!("Inference Test Finished");
}

// Main Function to Run Everything

fn main() {
    println!("Running Two-Layer Net Example");
    train_two_layer_net();
    println!("Finished");   
}