use burn::tensor::backend::Backend;  
use burn::tensor::Tensor;         
use burn::tensor::Distribution::Normal;

// Function to generate random sample points across Normal distribution on interval [-1,1]
pub fn gen_collocation_points<B: Backend>(
    device: &B::Device,
    m: usize,
) -> Tensor<B, 2> {
    Tensor::<B, 2>::random(
        [m, 1],
        Normal(0.0, 1.0),
        device,
    ).clamp(-1.0, 1.0)
}

fn source_function<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
    let pi = std::f32::consts::PI;
    let pi_sq = pi * pi;
    let sin_pix = x.clone().mul_scalar(pi).sin();
    sin_pix.mul_scalar(pi_sq)

}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{backend::Backend, Tensor};
    use burn::backend::NdArray;

    use crate::train::gen_collocation_points;
    #[test]
    fn test_random_sample_points() {
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();
        let m = 10000;

        // Generate random sample points
        let x = gen_collocation_points::<B>(&device, m);

        // Shape check
        let shape = x.dims();
        assert_eq!(
            shape,
            [m, 1],
            "Tensor shape mismatch: expected [{}, 1], got {:?}",
            m,
            shape
        );

        // Value in bounds check
        let data = x.to_data();
        let values: Vec<f32> = data.as_slice().unwrap().to_vec();

        for (i, &v) in values.iter().enumerate() {
            assert!(
                v >= -1.0 && v <= 1.0,
                "Value out of bounds at index {}: {:.4}",
                i,
                v
            );
        }

        // Statistics Check
        let mean: f32 = values.iter().copied().sum::<f32>() / values.len() as f32;
        assert!(
            mean.abs() < 0.1,
            "Mean too far from 0, got {:.4} (check distribution)",
            mean
        );

        // --- No NaN or Inf values ---
        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at index {}: {}", i, v);
        }
    }

    #[test]
    fn test_source_function() {
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();

        // Define some test points (Tensor of shape [N,1])
        let x_vals: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let x_values = Tensor::<B, 1>::from_floats(x_vals.as_slice(), &device).reshape([5, 1]);

        // Compute source function outputs
        let f_pred = source_function::<B>(&x_values).to_data().to_vec::<f32>().unwrap();

        // Compute true values manually
        let pi = std::f32::consts::PI;
        let pi_sq = pi * pi;
        let f_true: Vec<f32> = [-1.0, -0.5, 0.0, 0.5, 1.0]
            .iter()
            .map(|&x| pi_sq * (pi * x).sin())
            .collect();

        // Compare predicted vs true
        for (i, (pred, truth)) in f_pred.iter().zip(f_true.iter()).enumerate() {
            let diff = (pred - truth).abs();
            assert!(
                diff < 1e-5,
                "Mismatch at index {}: got {:.6}, expected {:.6}",
                i,
                pred,
                truth
            );
        }
    }
}