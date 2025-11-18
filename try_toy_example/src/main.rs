mod cheb_points;
mod interp;
mod utils;
mod model;
mod train;


// use cheb_points::*;
// use interp::*;
use train::*;

use burn::tensor::backend::Backend;     // Backend trait
use burn_autodiff::Autodiff;            // Autodiff tensor
// use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer, SgdConfig};    // optimizer
use burn::backend::NdArray;             // NdArray backend

// Choose a concrete backend type parameter
type B = Autodiff<NdArray>;             // for CPU

fn main() {
    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();
    let epochs = 500;
    let n = 100;
    let m = 100;
    let learning_rate = 0.00001;
    println!("Running Two-Layer Net");
    train_model::<B>(device, epochs, n, m, learning_rate);
    println!("Finished Running Two-Layer Net");
}
