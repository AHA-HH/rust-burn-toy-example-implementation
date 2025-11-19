use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

// Barycentric interpolation/evaluation using Chebyshev points in 1D
pub fn cheb_1d_interpolate<B: Backend>(
    device: &B::Device,
    e_points: &Tensor<B, 1>,
    f_values: &Tensor<B, 1>,
    c_points: &Tensor<B ,1>,
    b_weights: &Tensor<B, 1>
) -> Tensor<B, 1> {
    let eps = 1e-12;

    let m = e_points.dims()[0];
    let n = c_points.dims()[0];

    let eval_points = e_points.clone().reshape([m, 1]);
    let cheb_points = c_points.clone().reshape([1, n]);

    let diff = eval_points.clone() - cheb_points.clone();

    // Need to figure out a way to catch the 0 values produced by exact matches, to stop it blowing up
    // one way could be to resort to a for loop but i dont know if tensors support this behaviour
    // moving away from the tensor type to a vector type is not possible or we'll break the flow needed for the neural network to learn

    let inv_diff = diff.recip();

    let bary_weights = b_weights.clone().reshape([1, n]);
    let func_values = f_values.clone().reshape([1, n]);

    let numerator = (bary_weights.clone() * func_values.clone() * inv_diff.clone()).sum_dim(1);

    let denominator = (bary_weights.clone() * inv_diff.clone()).sum_dim(1);

    let interp = numerator / denominator;

    interp.reshape([m])
}

#[cfg(test)]
mod test {
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    use crate::{
        cheb_points::{gen_cheb_points, gen_barycentric_weights},
        interp::*
    };

    fn t3(x: f32) -> f32 { 4.0 * x * x * x - 3.0 * x }
    fn t4(y: f32) -> f32 { 8.0 * y * y * y * y - 8.0 * y * y + 1.0 }

    #[test]
    fn test_cheb_1d_interpolation() {
        let device = <B as Backend>::Device::default();
        let n = 10;

        let cheb_points = gen_cheb_points::<B>(&device, n);
        let bary_weights = gen_barycentric_weights::<B>(&device, n);

        let cheb_points_vec = cheb_points.to_data().to_vec::<f32>().unwrap();
        let f_values_vec: Vec<f32> = cheb_points_vec.iter().copied().map(t3).collect();
        let f_values = Tensor::<B, 1>::from_floats(f_values_vec.as_slice(), &device);

        let m = 20;

        let eval_points = gen_cheb_points::<B>(&device, m);
        let eval_points_vec = eval_points.to_data().to_vec::<f32>().unwrap();
        let true_values_vec: Vec<f32> = eval_points_vec.iter().copied().map(t3).collect();
        let true_values = Tensor::<B, 1>::from_floats(true_values_vec.as_slice(), &device);

        let interp_values = cheb_1d_interpolate(
            &device, 
            &eval_points, 
            &f_values, 
            &cheb_points, 
            &bary_weights
        );

        let abs_error = (interp_values.clone() - true_values.clone()).abs();
        let max_error = abs_error.max().to_data().to_vec::<f32>().unwrap()[0];
       
        assert!(
            max_error < 1e-5,
            "Max 1D interpolation error too high: {}",
            max_error
        );

    }
}