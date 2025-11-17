// Barycentric interpolation using Chebyshev points in 1D
pub fn cheb_1d_interpolate(
    eval_points: Vec<f32>, 
    values: Vec<f32>, 
    c_points: Vec<f32>, 
    c_weights: Vec<f32>) -> Vec<f32> {
    let npoints = eval_points.len();
    let nnodes = values.len();

    let mut numerator = vec![0.0; npoints];
    let mut denominator = vec![0.0; npoints];
    let mut exact_idx: Vec<Option<usize>> = vec![None; npoints];

    assert_eq!(nnodes, c_points.len());
    assert_eq!(nnodes, c_weights.len());
    
    for (j, &x_j) in c_points.iter().enumerate(){
        let w_j = c_weights[j];
        let f_j = values[j];
        for (i, &xeval) in eval_points.iter().enumerate(){
            let diff = xeval - x_j;
            
            if diff.abs() <= 1e-14 * (1.0_f32.max(xeval.abs())) {
                exact_idx[i] = Some(j);
                continue;
            }

            let inv_diff = 1.0 / diff;
            numerator[i] += w_j * f_j * inv_diff;
            denominator[i] += w_j * inv_diff;
        }
    }

    let mut result = vec![0.0; npoints];
    
    for i in 0..npoints {
        if let Some(j) = exact_idx[i] {
            result[i] = values[j];
        } else {
            result[i] = numerator[i] / denominator[i];
        }
    }

    result
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
    fn test_cheb_1d() {
        let n = 10;
        let nodes = gen_cheb_points(n);
        let weights = gen_barycentric_weights(n);

        let values: Vec<f32> = nodes.iter().copied().map(t3).collect();

        let m = 50;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let eval_points: Vec<f32> = (0..m)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        
        let interp_values = cheb_1d_interpolate(
            eval_points.clone(),
            values.clone(),
            nodes.clone(),
            weights.clone(),
        );

        let true_values: Vec<f32> = eval_points.iter().copied().map(t3).collect();

        let max_error = interp_values.iter()
            .zip(true_values.iter())
            .map(|(&interp, &true_val)| (interp - true_val).abs())
            .fold(0.0_f32, |a, b| a.max(b));

        assert!(max_error < 1e-5, "Max 1D interpolation error: {}", max_error);
    }

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