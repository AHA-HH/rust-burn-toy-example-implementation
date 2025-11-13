use burn::tensor::Tensor;

fn main() {
    // Simple example to verify installation:
    let device = burn::tensor::backend::ndarray::NdArrayDevice::default();
    let x = Tensor::<burn::tensor::backend::ndarray::NdArrayBackend<f32>, 1>::from_floats([1.0, 2.0, 3.0], &device);
    println!("{:?}", x);
}