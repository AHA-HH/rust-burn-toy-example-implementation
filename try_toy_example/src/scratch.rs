use burn::tensor::{Tensor, backend::Backend};

/// 2D barycentric interpolation using tensor operations.
/// Interpolates first along y for each x, then along x for each y.
pub fn cheb_2d_interpolate_tensor<B: Backend>(
    device: &B::Device,
    eval_x: &Tensor<B, 1>,   // shape [mx]
    eval_y: &Tensor<B, 1>,   // shape [my]
    values: &Tensor<B, 2>,   // shape [nx, ny]
    c_points_x: &Tensor<B, 1>,
    b_weights_x: &Tensor<B, 1>,
    c_points_y: &Tensor<B, 1>,
    b_weights_y: &Tensor<B, 1>,
) -> Tensor<B, 2> {
    let nx = c_points_x.dims()[0];
    let ny = c_points_y.dims()[0];
    let mx = eval_x.dims()[0];
    let my = eval_y.dims()[0];

    // --- Step 1: interpolate along y for each fixed x node ---
    let mut interp_y_results: Vec<Tensor<B, 1>> = Vec::with_capacity(nx);
    for jx in 0..nx {
        // slice row jx → shape [ny]
        let row_vals = values.clone().slice([jx..jx + 1]).reshape([ny]);
        let interp_y = crate::interp::cheb_1d_interpolate(
            device,
            eval_y,
            &row_vals,
            c_points_y,
            b_weights_y,
        ); // shape [my]
        interp_y_results.push(interp_y);
    }

    // stack results to get shape [nx, my]
    let temp_y = Tensor::stack(interp_y_results, 0);

    // --- Step 2: interpolate along x for each fixed y evaluation ---
    let mut result_cols: Vec<Tensor<B, 1>> = Vec::with_capacity(my);
    for iy in 0..my {
        // take column iy → shape [nx]
        let col_vals = temp_y.clone().slice([0..nx, iy..iy + 1]).reshape([nx]);
        let interp_x = crate::interp::cheb_1d_interpolate(
            device,
            eval_x,
            &col_vals,
            c_points_x,
            b_weights_x,
        ); // shape [mx]
        result_cols.push(interp_x);
    }

    // stack along second axis to get [mx, my]
    Tensor::stack(result_cols, 1)
}

#[cfg(test)]
mod test {
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    use crate::cheb_points::{gen_cheb_points, gen_barycentric_weights};
    use crate::interp::{cheb_1d_interpolate, cheb_2d_interpolate_tensor};

    fn t3(x: f32) -> f32 { 4.0 * x * x * x - 3.0 * x }
    fn t4(y: f32) -> f32 { 8.0 * y * y * y * y - 8.0 * y * y + 1.0 }

    #[test]
    fn test_cheb_2d_interpolation() {
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let nx = 10;
        let ny = 15;

        let x_nodes = gen_cheb_points::<B>(&device, nx);
        let y_nodes = gen_cheb_points::<B>(&device, ny);
        let lam_x = gen_barycentric_weights::<B>(&device, nx);
        let lam_y = gen_barycentric_weights::<B>(&device, ny);

        // Build 2D function values f(x_i, y_j)
        let mut values = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                values[i * ny + j] = t3(x_nodes.clone().to_data().to_vec::<f32>().unwrap()[i])
                    * t4(y_nodes.clone().to_data().to_vec::<f32>().unwrap()[j]);
            }
        }
        let values_tensor = Tensor::<B, 2>::from_floats(&values, &device).reshape([nx, ny]);

        // Evaluation grid
        let mx = 40;
        let my = 50;
        let eval_x = gen_cheb_points::<B>(&device, mx);
        let eval_y = gen_cheb_points::<B>(&device, my);

        // Interpolate
        let interp_vals = cheb_2d_interpolate_tensor(
            &device,
            &eval_x,
            &eval_y,
            &values_tensor,
            &x_nodes,
            &lam_x,
            &y_nodes,
            &lam_y,
        );

        // Compute ground truth
        let eval_x_vec = eval_x.to_data().to_vec::<f32>().unwrap();
        let eval_y_vec = eval_y.to_data().to_vec::<f32>().unwrap();
        let interp_vec = interp_vals.to_data().to_vec::<f32>().unwrap();

        let mut max_err = 0.0;
        for ix in 0..mx {
            for iy in 0..my {
                let true_val = t3(eval_x_vec[ix]) * t4(eval_y_vec[iy]);
                let interp_val = interp_vec[ix * my + iy];
                max_err = max_err.max((interp_val - true_val).abs());
            }
        }

        assert!(max_err < 1e-3, "Max 2D interpolation error too high: {}", max_err);
    }
}