use burn::tensor::backend::Backend;  
use burn::tensor::Tensor;         
use burn::tensor::Distribution::Normal;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn_autodiff::Autodiff;
use burn::module::{Module, AutodiffModule};
use ndarray::Array2;
use crate::interp::cheb_1d_interpolate;
use crate::cheb_points::*;
use crate::model::*;
use crate::utils::*;

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

// Function to force a function to learn, in this case u(x) = sin(pi * x)
pub fn source_function<B: Backend>(x: &Tensor<B, 2>) -> Tensor<B, 2> {
    let pi = std::f32::consts::PI;
    let pi_sq = pi * pi;
    let sin_pix = x.clone().mul_scalar(pi).sin();
    sin_pix.mul_scalar(pi_sq)
}

// Function to compute residual to prepare for loss function
pub fn compute_residual<B: Backend>(
    u_pred: &Tensor<B, 2>,
    x_cheb: Vec<f32>,
    d2_matrix: &Array2<f32>,
    x_rand: &Tensor<B, 2>,
    b_weights: Vec<f32>,
    device: &B::Device,
) -> Tensor<B, 2> {
    // let u_pred_vec = u_pred.to_data().to_vec::<f32>().unwrap();
    // let uxx_pred_vec = d2_matrix.dot(&ndarray::Array1::from(u_pred_vec));
    
    // let d2_tensor = Tensor::<B ,2>::from_floats(d2_matrix
    //     .as_slice().unwrap(), device).reshape([n, n]);

    // let uxx_pred = d2_tensor.matmul(u_pred.clone());

    // let uxx_interp = cheb_1d_interpolate(
    //     x_rand.to_data().to_vec::<f32>().unwrap(), 
    //     uxx_pred_vec.to_vec(), 
    //     x_cheb, 
    //     b_weights
    // );

    // let f_true = source_function::<B>(&x_rand);

    // let x_cheb_tensor = cheb_points_tensor::<B>(device, n);
    // let x_cheb_tensor = Tensor::<B, 1>::from_floats(x_cheb.as_slice(), device)
    // .reshape([n, 1]);

    // let f_true = source_function::<B>(&x_cheb);

    // let f_true = source_function::<B>(&x_cheb_tensor);

    // let residuals = - uxx_pred - f_true;

    // let uxx_interp_tensor = Tensor::<B, 1>::from_floats(&*uxx_interp, device)
    // .reshape([x_rand.dims()[0], 1]);
   
    // let residuals = uxx_interp_tensor.neg() - f_true;

    // residuals
    
    u_pred.clone().powf_scalar(2.0)
}

// Function to calculate loss using Clenshaw-Curtis method
pub fn compute_loss<B: Backend>(
    residuals: &Tensor<B, 2>,
    weights: &Vec<f32>,
    device: &B::Device,
) -> Tensor<B, 1> {
    // let weight_tensor = Tensor::<B, 2>::from_floats(weights.as_slice(), device)
    // .reshape([weights.len(), 1]);

    let weight_tensor = Tensor::<B, 1>::from_floats(weights.as_slice(), device)
    .reshape([weights.len(), 1]);

    let weighted_residuals = residuals.clone().powf_scalar(2.0) * weight_tensor;

    let loss = weighted_residuals.sum();

    loss
}

// Function to carry out training loop for neural network
pub fn train_model<B: Backend> (
    device: B::Device,
    epochs: usize,
    n: usize,
    m: usize,
    learning_rate: f32, // include an arg for hidden size of layer?
) {

    let (x_cheb, d1) = gen_cheb_diff_matrix(n);
    let d2 = get_cheb_diff_matrix_second(&d1);
    let b_weights = gen_barycentric_weights(n);

    let x_rand = gen_collocation_points::<Autodiff<B>>(&device, m);

    let cc_weights = gen_clenshaw_curtis_weights(m - 1);

    let mut model: TwoLayerNet<Autodiff<B>> = TwoLayerNetConfig {
        input_features: 1,
        hidden_features: 20,
        output_features: 1,
    }
    .init(&device);

    let mut optim = AdamConfig::new().init::<Autodiff<B>, TwoLayerNet<Autodiff<B>>>();

    println!("Training Started");
    for epoch in 0..epochs {
        let x_tensor = cheb_points_tensor::<Autodiff<B>>(&device, n).reshape([n, 1]);

        let u_pred = model.forward(x_tensor);
        
        let residuals = compute_residual::<Autodiff<B>>(
            &u_pred, 
            x_cheb.clone(), 
            &d2, 
            &x_rand, 
            b_weights.clone(), 
            &device,
        );

        let loss = compute_loss::<Autodiff<B>>(&residuals, &cc_weights, &device);

        let gradients = loss.backward();

        model = optim.step(
            learning_rate.into(),
            model.clone(),
            GradientsParams::from_grads(gradients, &model),
        );

        println!("Step {}: Loss: {:.4}", epoch + 1, loss.to_data());
    }

    println!("Training Finished");
    // println!("Testing Started");
    // let test_x: Vec<f32> = (0..100)
    // .map(|i| -1.0 + 2.0 * (i as f32 / 99.0))
    // .collect();

    // let test_input = Tensor::<Autodiff<B>, 1>::from_floats(test_x.as_slice(), &device)
    // .reshape([100, 1]);

    // let pred_y = model.forward(test_input.clone());

    // println!("x\ty_true\ty_pred");
    // let pred_data = pred_y.to_data().to_vec::<f32>().unwrap();

    // for (x, y_pred) in test_x.iter().zip(pred_data.iter()) {
    //     let y_true = (std::f32::consts::PI * x).sin();
    //     println!("{:.3}\t{:.3}\t{:.3}", x, y_true, y_pred);
    // }
    // println!("Testing Finished")
}


#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{backend::Backend, Tensor};
    use burn::backend::NdArray;

    use crate::train::{gen_collocation_points, source_function};
    use crate::cheb_points::*;

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

        // No NaN or Inf values
        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite value at index {}: {}", i, v);
        }
    }

    #[test]
    fn test_source_function() {
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();

        // Define some random test points
        let m = 10;
        let x_rand = gen_collocation_points(&device, m);

        // Compute true values functionally
        let f_true = source_function::<B>(&x_rand).to_data().to_vec::<f32>().unwrap();
        let x_vals = x_rand.to_data().to_vec::<f32>().unwrap();

        // Compute true values manually
        let pi = std::f32::consts::PI;
        let pi_sq = pi * pi;
        let f_manual: Vec<f32> = x_vals
            .iter()
            .map(|&x| pi_sq * (pi * x).sin())
            .collect();

        // Compare predicted vs true
        for (i, (truth, manual)) in f_true.iter().zip(f_manual.iter()).enumerate() {
            let diff = (truth - manual).abs();
            assert!(
                diff < 1e-5,
                "Mismatch at index {}: got {:.6}, expected {:.6}",
                i,
                truth,
                manual
            );
        }
    }

    // #[test]
    // fn test_residual_calculation() {
    //     type B = NdArray<f32>;
    //     let device = <B as Backend>::Device::default();

    //     let n = 20;
    //     let x_cheb = gen_cheb_points(n);
    //     let b_weights = gen_barycentric_weights(n);
    
    //     let (_x, d1) = gen_cheb_diff_matrix(n);

    //     let d2 = get_cheb_diff_matrix_second(&d1);
        
    //     let u_vals: Vec<f32> = x_cheb.clone();
    //     let u_vals_f32: Vec<f32> = u_vals.iter().map(|&v| v as f32).collect();
        
    //     let u_pred = Tensor::<B, 1>::from_floats(u_vals_f32.as_slice(), &device)
    //         .reshape([n, 1]);

    //     let m = 50;
    //     let x_rand = gen_collocation_points::<B>(&device, m);

    //     // let x_tensor = cheb_points_tensor(&device, n);

    //     let residuals = compute_residual::<B>(
    //         n,
    //         &u_pred,
    //         x_cheb.clone(),
    //         &d2,
    //         &x_rand,
    //         b_weights.clone(),
    //         &device,
    //     );

    //     let shape = residuals.dims();
    //     assert_eq!(shape, [m, 1], "Residual tensor shape should be [m, 1]");

    //     // No NaN or Inf values
    //     let vals = residuals.to_data().to_vec::<f32>().unwrap();
    //     assert!(
    //         vals.iter().all(|&v| v.is_finite()),
    //         "Residual tensor contains NaN or Inf"
    //     );

    //     // Ensure non-zero output
    //     let max_val = vals.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
    //     assert!(
    //         max_val > 0.0,
    //         "Residual tensor unexpectedly all zeros — interpolation may have failed"
    //     );

    //     println!("Residual interpolation shape: {:?}, max |val| = {:.3e}", shape, max_val);
    // }

    // #[test]
    // fn test_loss_calculation() {
    //     type B = NdArray<f32>;
    //     let device = <B as Backend>::Device::default();

    //     let n = 20;
    //     let x_cheb = gen_cheb_points(n);
    //     let b_weights = gen_barycentric_weights(n);
    
    //     let (_x, d1) = gen_cheb_diff_matrix(n);

    //     let d2 = get_cheb_diff_matrix_second(&d1);
        
    //     let u_vals: Vec<f32> = x_cheb.clone();
    //     let u_vals_f32: Vec<f32> = u_vals.iter().map(|&v| v as f32).collect();
        
    //     let u_pred = Tensor::<B, 1>::from_floats(u_vals_f32.as_slice(), &device)
    //         .reshape([n, 1]);

    //     let m = 50;
    //     let x_rand = gen_collocation_points::<B>(&device, m);

    //     let residuals = compute_residual::<B>(
    //         n,
    //         &u_pred,
    //         x_cheb.clone(),
    //         &d2,
    //         &x_rand,
    //         b_weights.clone(),
    //         &device,
    //     );
   
    //     let weights = gen_clenshaw_curtis_weights(m - 1);

    //     let loss = compute_loss::<B>(&residuals, &weights, &device);

    //     let loss_val = loss.to_data().to_vec::<f32>().unwrap()[0];

    //      assert_eq!(
    //     loss.dims(),
    //     [1],
    //     "Loss tensor should have shape [1], got {:?}",
    //     loss.dims()
    // );

    // // Ensure it's finite and non-negative
    // assert!(
    //     loss_val.is_finite() && loss_val >= 0.0,
    //     "Loss should be finite and non-negative, got {:.6}",
    //     loss_val
    // );

    // // The residual-based loss should be small
    // assert!(
    //     loss_val < 100.0,
    //     "Loss too large ({:.6}) — residual computation may be off",
    //     loss_val
    // );

    // println!(
    //     "Loss test passed: Loss = {:.6} (m={}, n={})",
    //     loss_val, m, n
    // );
    // }
}