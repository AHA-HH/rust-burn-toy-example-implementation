mod cheb_points;
mod interp;
mod utils;
mod model;
mod train;


use cheb_points::*;
// use interp::*;

use burn::tensor::backend::Backend;     // Backend trait
use burn_autodiff::Autodiff;            // Autodiff tensor
// use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer, SgdConfig};    // optimizer
use burn::backend::NdArray;             // NdArray backend

// Choose a concrete backend type parameter
type B = Autodiff<NdArray>;             // for CPU

fn main() {
    let device = <B as Backend>::Device::default();
}
