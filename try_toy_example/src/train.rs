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


// Function to force a function to learn, in this case u(x) = sin(pi * x)
pub fn source_function<B: Backend>(x: &Tensor<B, 1>) -> Tensor<B, 1> {
    let pi = std::f32::consts::PI;
    let pi_sq = pi * pi;
    let sin_pix = x.clone().mul_scalar(pi).sin();
    sin_pix.mul_scalar(pi_sq)
}

// Function to compute residual to prepare for loss function
pub fn compute_residual<B: Backend>(
    u_pred: &Tensor<B, 1>,
    x_cheb_points: &Tensor<B, 1>,
    d2_matrix: &Tensor<B, 2>,
    x_eval_points: &Tensor<B, 1>,
    b_weights: &Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let n = x_cheb_points.dims()[0];

    let u_pred_col = u_pred.clone().reshape([n, 1]);

    let uxx_pred_tensor = d2_matrix.clone().matmul(u_pred_col.clone());

    let uxx_pred = uxx_pred_tensor.reshape([n]);

    let uxx_interp = cheb_1d_interpolate(
        device, 
        x_eval_points, 
        &uxx_pred, 
        x_cheb_points, 
        b_weights
    );

    let f_true = source_function::<B>(&x_eval_points);

    let residuals = uxx_interp.neg() - f_true;

    residuals

    // u_pred.clone().powf_scalar(2.0)
}

// Function to calculate loss using Clenshaw-Curtis method
pub fn compute_loss<B: Backend>(
    residuals: &Tensor<B, 1>,
    weights: &Tensor<B, 1>,
    // device: &B::Device,
) -> Tensor<B, 1> {
    let weighted_residuals = residuals.clone().powf_scalar(2.0) * weights.clone();

    let loss = weighted_residuals.sum();

    loss
}

// Function to carry out training loop for neural network, include an arg for hidden size of layer?
pub fn train_model<B: Backend> (
    device: B::Device,
    epochs: usize,
    n: usize,
    m: usize,
    learning_rate: f32, 
) {
    let x_cheb = gen_cheb_points::<Autodiff<B>>(&device, n);
    let b_weights = gen_barycentric_weights::<Autodiff<B>>(&device, n);

    let d1 = gen_cheb_diff_matrix::<Autodiff<B>>(&device, &x_cheb.clone());
    let d2 = get_cheb_diff_matrix_second::<Autodiff<B>>(&d1);

    let x_eval = gen_cheb_points::<Autodiff<B>>(&device, m);

    let cc_weights = gen_clenshaw_curtis_weights::<Autodiff<B>>(&device, m - 1);
    let mut model: TwoLayerNet<Autodiff<B>> = TwoLayerNetConfig {
        input_features: 1,
        hidden_features: 20,
        output_features: 1,
    }
    .init(&device);

    let mut optim = AdamConfig::new().init::<Autodiff<B>, TwoLayerNet<Autodiff<B>>>();
    println!("Training Started");
    for epoch in 0..epochs {
        let u_pred = model.forward(x_cheb.clone());

        let residuals = compute_residual::<Autodiff<B>>(
            &u_pred, 
            &x_cheb.clone(), 
            &d2, 
            &x_eval, 
            &b_weights.clone(), 
            &device,
        );

        let loss = compute_loss::<Autodiff<B>>(&residuals, &cc_weights);
 
        let gradients = loss.backward();

        model = optim.step(
            learning_rate.into(),
            model.clone(),
            GradientsParams::from_grads(gradients, &model),
        );

        println!("Step {}: Loss: {:.4}", epoch + 1, loss.to_data());
    }

    println!("Training Finished");
    println!("Testing Started");
    let test_x: Vec<f32> = (0..100)
    .map(|i| -1.0 + 2.0 * (i as f32 / 99.0))
    .collect();

    let test_input = Tensor::<Autodiff<B>, 1>::from_floats(&*test_x, &device);

    let pred_y = model.forward(test_input.clone());

    println!("x\ty_true\ty_pred");
    let pred_data = pred_y.to_data().to_vec::<f32>().unwrap();

    for (x, y_pred) in test_x.iter().zip(pred_data.iter()) {
        let y_true = (std::f32::consts::PI * x).sin();
        println!("{:.3}\t{:.3}\t{:.3}", x, y_true, y_pred);
    }
    println!("Testing Finished")
}


#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{backend::Backend, Tensor};
    use burn::backend::NdArray;

    use crate::train::*;
    use crate::cheb_points::*;

    #[test]
    fn test_source_function() {
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();

        // Define some random test points
        let m = 10;
        let x_eval = gen_cheb_points::<B>(&device, m);

        // Compute true values functionally
        let f_true = source_function::<B>(&x_eval).to_data().to_vec::<f32>().unwrap();
        let x_vals = x_eval.to_data().to_vec::<f32>().unwrap();

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

    #[test]
    fn test_residual_calculation() {
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();

        let n = 20;
        let x_cheb = gen_cheb_points::<B>(&device, n);
        let b_weights = gen_barycentric_weights::<B>(&device, n);
    
        let d1 = gen_cheb_diff_matrix::<B>(&device, &x_cheb.clone());

        let d2 = get_cheb_diff_matrix_second(&d1);
        
        let u_pred = x_cheb.clone();

        let m = 50;
        let x_eval = gen_cheb_points::<B>(&device, m);

        // let x_tensor = cheb_points_tensor(&device, n);

        let residuals = compute_residual::<B>(
            &u_pred,
            &x_cheb.clone(),
            &d2,
            &x_eval,
            &b_weights.clone(),
            &device,
        );

        let shape = residuals.dims();
        assert_eq!(shape, [m], "Residual tensor shape should be [m], got {:?}", shape);

        // No NaN or Inf values
        let vals = residuals.to_data().to_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|&v| v.is_finite()),
            "Residual tensor contains NaN or Inf"
        );

        // Ensure non-zero output
        let max_val = vals.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        assert!(
            max_val > 0.0,
            "Residual tensor unexpectedly all zeros — interpolation may have failed"
        );

        println!("Residual interpolation shape: {:?}, max |val| = {:.3e}", shape, max_val);
    }

    #[test]
    fn test_loss_calculation() {
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();

        let n = 20;
        let x_cheb = gen_cheb_points::<B>(&device, n);
        let b_weights = gen_barycentric_weights::<B>(&device, n);
    
        let d1 = gen_cheb_diff_matrix::<B>(&device, &x_cheb.clone());

        let d2 = get_cheb_diff_matrix_second::<B>(&d1);
        
        let u_pred = x_cheb.clone();

        let m = 50;
        let x_eval = gen_cheb_points::<B>(&device, m);

        let residuals = compute_residual::<B>(
            &u_pred,
            &x_cheb.clone(),
            &d2,
            &x_eval,
            &b_weights.clone(),
            &device,
        );
   
        let weights = gen_clenshaw_curtis_weights::<B>(&device, m - 1);

        let loss = compute_loss::<B>(&residuals, &weights);

        let loss_val = loss.to_data().to_vec::<f32>().unwrap()[0];

         assert_eq!(
        loss.dims(),
        [1],
        "Loss tensor should have shape [1], got {:?}",
        loss.dims()
    );

    // Ensure it's finite and non-negative
    assert!(
        loss_val.is_finite() && loss_val >= 0.0,
        "Loss should be finite and non-negative, got {:.6}",
        loss_val
    );

    // The residual-based loss should be small
    assert!(
        loss_val < 100.0,
        "Loss too large ({:.6}) — residual computation may be off",
        loss_val
    );

    println!(
        "Loss test passed: Loss = {:.6} (m={}, n={})",
        loss_val, m, n
    );
    }
}