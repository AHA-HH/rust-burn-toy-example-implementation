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

    let abs_diff = diff.clone().abs();

    let inv_diff = diff.recip();

    let bary_weights = b_weights.clone().reshape([1, n]);
    let func_values = f_values.clone().reshape([1, n]);

    let numerator = (bary_weights.clone() * func_values.clone() * inv_diff.clone()).sum_dim(1);

    let denominator = (bary_weights.clone() * inv_diff.clone()).sum_dim(1);

    let interp = numerator / denominator;

    interp.reshape([m])
}

// Barycentric interpolation using Chebyshev points in 2D
pub fn cheb_2d_interpolate_tensor(
    eval_x: Vec<f32>,
    eval_y: Vec<f32>,
    values: Vec<Vec<f32>>,
    c_points_x: Vec<f32>,
    c_weights_x: Vec<f32>,
    c_points_y: Vec<f32>,
    c_weights_y: Vec<f32>) -> Vec<Vec<f32>> {

    let nx_eval = eval_x.len();
    let ny_eval = eval_y.len();
    let nx_nodes = c_points_x.len();
    let ny_nodes = c_points_y.len();


    assert_eq!(values.len(), nx_nodes, "values rows must equal x_nodes");
    for row in values.iter() {
        assert_eq!(row.len(), ny_nodes, "each values row must equal y_nodes");
    }
    // assert_eq!(x_weights.len(), nx_nodes);
    // assert_eq!(y_weights.len(), ny_nodes);
   
   // for each x, interpolate along y direction
    let mut temp: Vec<Vec<f32>> = Vec::with_capacity(nx_nodes);
    for jx in 0..nx_nodes {
        let row_vals = values[jx].clone(); 
        let interp_y = cheb_1d_interpolate(
            eval_y.clone(),    
            row_vals,      
            c_points_y.clone(),
            c_weights_y.clone(),
        ); 
        temp.push(interp_y);
    }

    // for each y, interpolate along x direction
    let mut result = vec![vec![0.0; ny_eval]; nx_eval];

    for iy in 0..ny_eval {
        let mut col = vec![0.0; nx_nodes];
        for jx in 0..nx_nodes {
            col[jx] = temp[jx][iy];
        }
        let interp_x = cheb_1d_interpolate(
            eval_x.clone(),
            col,
            c_points_x.clone(),
            c_weights_x.clone(),
        );

        for ix in 0..nx_eval {
            result[ix][iy] = interp_x[ix];
        }
    }

    result
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        cheb_points::{gen_cheb_points, gen_barycentric_weights},
        interp::{cheb_1d_interpolate, cheb_2d_interpolate_tensor}
    };

    fn t3(x: f32) -> f32 { 4.0 * x * x * x - 3.0 * x }
    fn t4(y: f32) -> f32 { 8.0 * y * y * y * y - 8.0 * y * y + 1.0 }

    #[test]
    fn test_cheb_2d() {
        let nx = 10;
        let ny = 15;
        let x_nodes = gen_cheb_points(nx);
        let y_nodes = gen_cheb_points(ny);
        let lam_x   = gen_barycentric_weights(nx);
        let lam_y   = gen_barycentric_weights(ny);

        let mut values2d = vec![vec![0.0; ny]; nx];
        for jx in 0..nx {
            for ky in 0..ny {
                values2d[jx][ky] = t3(x_nodes[jx]) * t4(y_nodes[ky]);
            }
        }

        let mx = 300;
        let my = 280;

        let mut rng = ChaCha8Rng::seed_from_u64(24);

        let eval_x: Vec<f32> = (0..mx)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let eval_y: Vec<f32> = (0..my)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let out = cheb_2d_interpolate_tensor(
            eval_x.clone(),
            eval_y.clone(),
            values2d,
            x_nodes.clone(),
            lam_x.clone(),
            y_nodes.clone(),
            lam_y.clone(),
        ); 

        let mut max_error = 0.0_f32;
        for ix in 0..mx {
            for iy in 0..my {
                let true_val = t3(eval_x[ix]) * t4(eval_y[iy]);
                let err = (out[ix][iy] - true_val).abs();
                if err > max_error {
                    max_error = err;
                }
            }
        }

        assert!(max_error < 1e-5, "Max 2D interpolation error: {}", max_error);
    }
}