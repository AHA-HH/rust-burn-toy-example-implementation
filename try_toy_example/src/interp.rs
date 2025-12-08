use burn::tensor::{Tensor, Shape, Bool};
use burn::tensor::backend::Backend;


// Barycentric interpolation/evaluation using Chebyshev points in 1D
// error is too high, due to either f32 or rounding errors
// further investigation into how interpolation is working
// as this could affect how neural network is learning down the line
pub fn cheb_1d_interpolate<B: Backend>(
    device: &B::Device,
    e_points: &Tensor<B, 1>,
    f_values: &Tensor<B, 1>,
    c_points: &Tensor<B ,1>,
    b_weights: &Tensor<B, 1>
) -> Tensor<B, 1> {
    let eps = 1e-6;

    let m = e_points.dims()[0];
    let n = c_points.dims()[0];

    let eval_points = e_points.clone().reshape([m, 1]);
    let cheb_points = c_points.clone().reshape([1, n]);

    let diff = eval_points.clone() - cheb_points.clone();

    let zero_mask = diff.clone().abs().lower_equal_elem(eps);
    
    let ones_data = vec![1.0; m * n];
    let ones_tensor = Tensor::<B, 1>::from_floats(&*ones_data, device).reshape([m, n]);

    let safe_diff = diff.clone().mask_where(zero_mask.clone(), ones_tensor);  

    let inv_diff = safe_diff.clone().recip();

    let bary_weights = b_weights.clone().reshape([1, n]);
    let func_values = f_values.clone().reshape([1, n]);

    let numerator = (bary_weights.clone() * func_values.clone() * inv_diff.clone()).sum_dim(1);

    let denominator = (bary_weights.clone() * inv_diff.clone()).sum_dim(1);

    let interp = numerator / denominator;

    let mask_f = zero_mask.clone().float();
    let func_match = mask_f.matmul(func_values.reshape([n, 1]));
    let row_mask = zero_mask.clone().any_dim(1);
    let corrected_interp = interp.mask_where(row_mask.clone(), func_match);

    corrected_interp.reshape([m])
}

// Barycentric interpolation using Chebyshev points in 2D
// pub fn cheb_2d_interpolate_tensor<B: Backend>(
//     device: &B::Device,
//     eval_x: &Tensor<B, 1>,
//     eval_y: &Tensor<B, 1>,
//     values: &Tensor<B, 2>,
//     c_points_x: &Tensor<B, 1>,
//     b_weights_x: &Tensor<B, 1>,   
//     c_points_y: &Tensor<B, 1>, 
//     b_weights_y: &Tensor<B, 1>
// ) -> Tensor<B, 2> {

//     let nx = c_points_x.dims()[0];
//     let ny = c_points_y.dims()[0];
//     let mx = eval_x.dims()[0];
//     let my = eval_y.dims()[0];

//     let mut interp_y_results: Vec<Tensor<B, 1>> = Vec::with_capacity(nx);
//     for jx in 0..nx {
//         // select row [jx, :]
//         let row_vals = values.clone().slice([jx..jx + 1, 0..ny]).reshape([ny]);
//         let interp_y = crate::interp::cheb_1d_interpolate(
//             device,
//             eval_y,
//             &row_vals,
//             c_points_y,
//             b_weights_y,
//         ); // shape [my]
//         interp_y_results.push(interp_y);
//     }

//     // Stack to shape [nx, my]
//     let temp_y: Tensor<B, 2> = Tensor::stack(interp_y_results, 0);

//     // Interpolate along x for each eval_y point
//     let mut interp_x_results: Vec<Tensor<B, 1>> = Vec::with_capacity(my);
//     for iy in 0..my {
//         // select column [:, iy]
//         let col_vals = temp_y.clone().slice([0..nx, iy..iy + 1]).reshape([nx]);
//         let interp_x = crate::interp::cheb_1d_interpolate(
//             device,
//             eval_x,
//             &col_vals,
//             c_points_x,
//             b_weights_x,
//         ); // shape [mx]
//         interp_x_results.push(interp_x);
//     }

//     // Stack along axis 1 to get shape [mx, my]
//     Tensor::stack(interp_x_results, 1)
// }

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
            max_error < 1e-3,
            "Max 1D interpolation error too high: {}",
            max_error
        );
    }

    // #[test]
    // fn test_cheb_2d_interpolation() {
    //     let device = <B as burn::tensor::backend::Backend>::Device::default();

    //     let nx = 10;
    //     let ny = 15;

    //     let x_nodes = gen_cheb_points::<B>(&device, nx);
    //     let y_nodes = gen_cheb_points::<B>(&device, ny);
    //     let lam_x = gen_barycentric_weights::<B>(&device, nx);
    //     let lam_y = gen_barycentric_weights::<B>(&device, ny);

    //     // Convert nodes to vectors once
    //     let x_nodes_vec = x_nodes.to_data().to_vec::<f32>().unwrap();
    //     let y_nodes_vec = y_nodes.to_data().to_vec::<f32>().unwrap();

    //     // Build 2D function values f(x_i, y_j)
    //     let mut values = vec![0.0; nx * ny];
    //     for i in 0..nx {
    //         for j in 0..ny {
    //             values[i * ny + j] = t3(x_nodes_vec[i]) * t4(y_nodes_vec[j]);
    //         }
    //     }
    //     let values_tensor = Tensor::<B, 2>::from_floats(&*values, &device).reshape([nx, ny]);

    //     // Evaluation grid
    //     let mx = 40;
    //     let my = 50;
    //     let eval_x = gen_cheb_points::<B>(&device, mx);
    //     let eval_y = gen_cheb_points::<B>(&device, my);

    //     // Interpolate
    //     let interp_vals = cheb_2d_interpolate_tensor(
    //         &device,
    //         &eval_x,
    //         &eval_y,
    //         &values_tensor,
    //         &x_nodes,
    //         &lam_x,
    //         &y_nodes,
    //         &lam_y,
    //     );

    //     // Compute and compare
    //     let eval_x_vec = eval_x.to_data().to_vec::<f32>().unwrap();
    //     let eval_y_vec = eval_y.to_data().to_vec::<f32>().unwrap();
    //     let interp_vec = interp_vals.to_data().to_vec::<f32>().unwrap();

    //     let mut max_err: f32 = 0.0;
    //     for ix in 0..mx {
    //         for iy in 0..my {
    //             let true_val = t3(eval_x_vec[ix]) * t4(eval_y_vec[iy]);
    //             let interp_val = interp_vec[ix * my + iy];
    //             max_err = max_err.max((interp_val - true_val).abs());
    //         }
    //     }

    //     assert!(
    //         max_err < 1e-12,
    //         "Max 2D interpolation error too high: {}",
    //         max_err
    //     );
    // }
}