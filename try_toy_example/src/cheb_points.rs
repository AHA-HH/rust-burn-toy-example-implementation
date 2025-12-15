use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

// Function to generate Chebyshev points of second kind
pub fn gen_cheb_points<B: Backend>(device: &B::Device, n: usize) -> Tensor<B, 1> {
    let pi = std::f64::consts::PI;
    let points: Vec<f64> = (0..n).map(|j| {
        let theta = ((n - 1 - j) as f64) * pi / ((n - 1) as f64);
        theta.cos()
    })
    .collect();

    // println!("Vector {:#?}", points);

    let tensor = Tensor::<B, 1>::from_data(points.as_slice(), device);
    // println!("Tensor: {:#?}", tensor);
    tensor
}

// Function to compute weights for Barycentric interpolation
pub fn gen_barycentric_weights<B: Backend>(device: &B::Device, n: usize) -> Tensor<B, 1> {
    let mut weights = vec![1.0; n];
    weights[0] = 0.5;
    weights[n-1] = 0.5;
    
    for j in 0..n {
        if j % 2 == 1 {
            weights[j] = -weights[j];
        }
    }

    Tensor::<B, 1>::from_data(weights.as_slice(), device)
}

// Function to generate Clenshaw-Curtis quadrature weights
pub fn gen_clenshaw_curtis_weights<B: Backend>(device: &B::Device, n: usize) -> Tensor<B, 1> {
    assert!(n >= 1, "n must be at least 1");
    
    let n_intervals = n + 1;
    let n_f = n_intervals as f64;


    let mut w = vec![0.0; n_intervals];
    let m = (n_f / 2.0).floor() as usize;

    for k in 0..=n as usize {
        let k_f = k as f64;
        let mut sum = 0.0;
        for j in 1..=m {
            let j_f = j as f64;
            let bj = if (j_f) == n_f / 2.0 {0.5} else {1.0};
            sum += bj * (2.0 * j_f * std::f64::consts::PI * k_f / n_f).cos() / (4.0 * (j_f * j_f) - 1.0);
        }
        w[k] = (2.0 / n_f) * (1.0 - sum);
        }
    
    Tensor::<B, 1>::from_data(w.as_slice(), device)
}

// Function to generate Chebyshev differentiation matrix; Reference: Spectral Methods in MATLAB Tregethen Ch.6 p.53
// maybe just use descending points, if this becomes too much of a pain to track
pub fn gen_cheb_diff_matrix<B: Backend>(device: &B::Device, x: &Tensor<B, 1>) -> Tensor<B, 2> {
    let x_vec = x.to_data().to_vec::<f64>().unwrap();
    let n = x_vec.len();
    
    // c_i: 2 for endpoints (0 or N), 1 otherwise
    let mut c = vec![1.0; n];
    c[0] = 2.0;
    c[n - 1] = 2.0;

    let mut d = vec![0.0; n * n];

    // Off-diagonal entries
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                d[i * n + j] = (c[i] / c[j]) * sign / (x_vec[j] - x_vec[i]); // reverse sign to match ascending Chebyshev points x[j] - x[i]
            }
        }
    }

    // Diagonal entries
    for i in 1..(n - 1) {
        d[i * n + i] = x_vec[i] / (2.0 * (1.0 - x_vec[i] * x_vec[i])); // change sign on diagonal to match ascending Chebyshev points
    }

    // Endpoints
    let n_f = (n - 1) as f64;
    let endpoint_val = (2.0 * n_f * n_f + 1.0) / 6.0; // To match ascending Chebyshev points
    d[0] = endpoint_val;
    d[n * n - 1] = -endpoint_val;

    // Convert to tensor
    Tensor::<B, 1>::from_data(d.as_slice(), device).reshape([n, n])
}

// Function to get the second Chebyshev differentiation matrix
pub fn get_cheb_diff_matrix_second<B: Backend>(d1: &Tensor<B, 2>) -> Tensor<B, 2> {
    d1.clone().matmul(d1.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use burn::backend::NdArray;

    type B = NdArray<f64>;

    #[test]
    fn test_cheb_points() {
        let device = <B as Backend>::Device::default();
        let n = 5;
        let points = gen_cheb_points::<B>(&device, n);
        let points_vec = points.to_data().to_vec::<f64>().unwrap();

        for (j, &x_j) in points_vec.iter().enumerate() {
            println!(" x_{} = {:.17}", j, x_j);
            let expected = ((n - 1 - j) as f64 * std::f64::consts::PI / (n - 1) as f64).cos();
            assert_relative_eq!(x_j, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_barycentric_weights() {
        let device = <B as Backend>::Device::default();

        for n in [2, 3, 4, 5, 8] {
            let w = gen_barycentric_weights::<B>(&device, n);
            let w_vec = w.to_data().to_vec::<f64>().unwrap();

            assert_eq!(w_vec.len(), n, "Length mismatch for n = {}", n);

            // Endpoint weights
            assert!((w_vec[0].abs() - 0.5).abs() < 1e-6, "First weight incorrect");
            assert!((w_vec[n - 1].abs() - 0.5).abs() < 1e-6, "Last weight incorrect");

            // Alternating signs
            for j in 1..n {
                assert!(
                    (w_vec[j] * w_vec[j - 1]) < 0.0,
                    "Weights do not alternate sign at indices {} and {} for n = {}: {:?}",
                    j - 1, j, n, w_vec
                );
            }
        }
    }

    #[test]
    fn test_clenshaw_curtis() {
        let device = <B as Backend>::Device::default();

        for points in [2, 4, 8, 16] {
            let w = gen_clenshaw_curtis_weights::<B>(&device, points);
            let w_vec = w.to_data().to_vec::<f64>().unwrap();

            let total: f64 = w_vec.iter().sum();
            // Check sum of weights is equal to 2
            assert!(
                (total - 2.0).abs() <= 1e-6,
                "Sum of weights for N = {} incorrect: got {}",
                points,
                total
            );
        }
    }

    #[test]
    fn test_cheb_diff_matrix() {
        let device = <B as Backend>::Device::default();
        let n = 5;

        let x_tensor = gen_cheb_points::<B>(&device, n);
        let d_tensor = gen_cheb_diff_matrix::<B>(&device, &x_tensor);

        let x_vec = x_tensor.to_data().to_vec::<f64>().unwrap();
        let d_vec = d_tensor.to_data().to_vec::<f64>().unwrap();

        // Reshape d_vec to 2D Vec<Vec<f64>> for easier logic reuse
        let mut d = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                d[i][j] = d_vec[i * n + j];
            }
        }

        assert_eq!(x_vec.len(), n);
        assert_eq!(d.len(), n);
        assert!(d.iter().all(|row| row.len() == n));

        // Derivative of constant function -> 0
        let u_const = vec![1.0; n];
        let du_const: Vec<f64> = d.iter()
            .map(|row| row.iter().zip(&u_const).map(|(dij, uj)| dij * uj).sum::<f64>())
            .collect();

        for (i, val) in du_const.iter().enumerate() {
            assert!(val.abs() < 1e-5, "Constant derivative not ~0 at index {}", i);
        }

        // Derivative of a linear function -> 1 or -1 depending on orientation
        let du_linear: Vec<f64> = d.iter()
            .map(|row| row.iter().zip(&x_vec).map(|(dij, xj)| dij * xj).sum::<f64>())
            .collect();

        for (i, val) in du_linear.iter().enumerate() {
            assert!((val + 1.0).abs() < 1e-5, "Derivative of x not ~1 at index {}", i);
        }

        // Endpoint diagonals symmetry
        assert!(
            (d[0][0] + d[n - 1][n - 1]).abs() < 1e-5,
            "Endpoint diagonals not symmetric"
        );

        // NaN / Inf checks
        for i in 0..n {
            for j in 0..n {
                assert!(!d[i][j].is_nan(), "NaN at ({}, {})", i, j);
                assert!(d[i][j].is_finite(), "Inf at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_matrix_multiplication() {
        let device = <B as Backend>::Device::default();
        let n = 3;

        let x_tensor = gen_cheb_points::<B>(&device, n);
        let d_tensor = gen_cheb_diff_matrix::<B>(&device, &x_tensor);
        let d2_tensor = get_cheb_diff_matrix_second(&d_tensor);

        let d2_vec = d2_tensor.to_data().to_vec::<f64>().unwrap();

        // Target comparison matrix
        let target = vec![
            1.0, -2.0, 1.0,
            1.0, -2.0, 1.0,
            1.0, -2.0, 1.0,
        ];

        let max_err = d2_vec.iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(max_err < 1e-3, "D² does not match expected pattern, max err = {:.3e}", max_err);
    }
}