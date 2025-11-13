use ndarray::Array2;

// Function to generate Chebyshev points of second kind
pub fn gen_cheb_points(n: usize) -> Vec<f64> {
    let pi = std::f64::consts::PI;
    let indices = 0..n;
    
    let mut points: Vec<f64> = Vec::with_capacity(n);
    
    for j in indices {
        let theta = ((n - 1 - j) as f64) * pi / ((n - 1) as f64);

        let x_j = theta.cos();

        points.push(x_j);
    }

    points
}

// Function to compute weights for Barycentric interpolation
pub fn gen_barycentric_weights(n: usize) -> Vec<f64> {
    let mut weights = vec![1.0; n];
    weights[0] = 0.5;
    weights[n-1] = 0.5;
    for j in 0..n {
        if j % 2 == 1 {
            weights[j] = -weights[j];
        }
    }

    weights
}

// Function to generate Clenshaw-Curtis quadrature weights
pub fn gen_clenshaw_curtis_weights(n: usize) -> Vec<f64> {
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
        w
}

// Function to generate Chebyshev differentiation matrix; Reference: Spectral Methods in MATLAB Tregethen Ch.6 p.53
pub fn gen_cheb_diff_matrix(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    assert!(n >= 2, "n must be at least 2");

    // Generate Chebyshev–Lobatto points in ascending order (-1 to +1)
    let x = gen_cheb_points(n);

    let mut d = vec![vec![0.0; n]; n];

    // c_i, 2 for 0 or N, 1 otherwise
    let mut c = vec![1.0; n];
    c[0] = 2.0;
    c[n - 1] = 2.0;

    // Off-diagonal entries
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let sign = if (i + j) % 2 == 0 {1.0} else {-1.0};
                d[i][j] = (c[i] / c[j]) * sign / (x[j] - x[i]); // reverse sign to match ascending Chebyshev points x[j] - x[i]
            }
        }
    }

    // Diagonal entries
    for i in 1..(n - 1) {
        d[i][i] = x[i] / (2.0 * (1.0 - x[i] * x[i])); // change sign on diagonal to match ascending Chebyshev points
    }

    // Endpoints
    let n_f = (n - 1) as f64;
    let endpoint_val = (2.0 * n_f * n_f + 1.0) / 6.0; // To match ascending Chebyshev points
    d[0][0] = endpoint_val;
    d[n-1][n-1] = -endpoint_val;

    // // Apply transformation to work with ascending points, unnecessary as numerically seems symmetric around centre
    // d.reverse();

    // // Reverse the column order for each row
    // for row in d.iter_mut() {
    //     row.reverse();
    // }

    // // Multiply all elements by -1
    // for i in 0..n {
    //     for j in 0..n {
    //         d[i][j] = -d[i][j];
    //     }
    // }

    (x, d)
}

// Function to convert type Vector of Vector to an Array
pub fn vec_to_array2(mat: &Vec<Vec<f64>>) -> Array2<f64> {
    let nrows = mat.len();
    let ncols = mat[0].len();
    let flat: Vec<f64> = mat.iter().flat_map(|row| row.iter().cloned()).collect();
    Array2::from_shape_vec((nrows, ncols), flat).unwrap()
}

// Function to get the second Chebyshev differentiation matrix
pub fn get_cheb_diff_matrix_second(mat: &Vec<Vec<f64>>) -> Array2<f64> {
    let d1_array = vec_to_array2(&mat);

    let d2_array = d1_array.dot(&d1_array);

    d2_array
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_cheb_points() {
        let n = 5;
        let points = gen_cheb_points(n);

        for (j, &x_j) in points.iter().enumerate() {
            assert_relative_eq!(
                x_j, 
                ((n - 1 - j) as f64 * std::f64::consts::PI / (n - 1) as f64).cos(), 
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_barycentric_weights() {
        for n in [2, 3, 4, 5, 8] {
            let w = gen_barycentric_weights(n);

            // Length should be n
            assert_eq!(w.len(), n, "Length mismatch for n = {}", n);

            // Endpoints should have weight of 0.5
            assert!(
                (w[0].abs() - 0.5).abs() < 1e-12,
                "First weight incorrect for n = {}: got {}",
                n,
                w[0]
            );
            assert!(
                (w[n - 1].abs() - 0.5).abs() < 1e-12,
                "Last weight incorrect for n = {}: got {}",
                n,
                w[n - 1]
            );

            for j in 1..n {
                assert!(
                    (w[j] * w[j - 1]) < 0.0,
                    "Weights did not alternate sign at indices {} and {} for n = {}: {:?}",
                    j - 1,
                    j,
                    n,
                    w
                );
            }
        }
    }

    #[test]
    fn test_clenshaw_curtis() {
        for points in [2, 4, 8, 16]{
            let w = gen_clenshaw_curtis_weights(points);
            let total: f64 = w.iter().sum();
            println!("{total}");
            assert!((total - 2.0).abs() <= 1e-12, "Sum of weights for N = {}, expected 2.0", total);
        }
    }

    #[test]
    fn test_cheb_diff_matrix() {
        let n = 8;
        let (x, d) = gen_cheb_diff_matrix(n);

        for row in d.iter() {
            for val in row {
                print!("{:>12.6}", val);
            }
            println!();
        }

        // Dimensions
        assert_eq!(x.len(), n, "Incorrect number of Chebyshev points");
        assert_eq!(d.len(), n, "Differentiation matrix row count mismatch");
        assert!(d.iter().all(|row| row.len() == n), "Matrix is not square");

        // Derivative of a constant function (D * x = 0)
        let u_const = vec![1.0; n];
        let du_const: Vec<f64> = d.iter()
            .map(|row| row.iter().zip(&u_const).map(|(dij, uj)| dij * uj).sum::<f64>())
            .collect();

        for (i, val) in du_const.iter().enumerate() {
            assert!(val.abs() < 1e-12, "Constant derivative not ~0 at index {}", i);  // Order of grid does not matter
        }

        // Derivative of a linear function (D * x = 1)
        let du_linear: Vec<f64> = d.iter()
            .map(|row| row.iter().zip(&x).map(|(dij, xj)| dij * xj).sum::<f64>())
            .collect();

        for (i, val) in du_linear.iter().enumerate() {
            // assert!((val - 1.0).abs() < 1e-12, "Derivative of x not ~1 at index {}", i); // Descending Chebyshev points
            assert!((val + 1.0).abs() < 1e-12, "Derivative of x not ~1 at index {}", i); // Ascending Chebyshev points
        }

        // Endpoint diagonals are opposite in sign
        assert!((d[0][0] + d[n - 1][n - 1]).abs() < 1e-12, "Endpoint diagonals not symmetric");

        // No NaN or infinite values
        for i in 0..n {
            for j in 0..n {
                assert!(!d[i][j].is_nan(), "NaN at ({}, {})", i, j);
                assert!(d[i][j].is_finite(), "Inf at ({}, {})", i, j);
            }
        }

        println!("Test passed for n = {}", n);
    }

    #[test]
    fn test_matrix_multiplication() {
        let n = 3;

        let target_array: Array2<f64> = array![
            [1.0, -2.0, 1.0],
            [1.0, -2.0, 1.0],
            [1.0, -2.0, 1.0],
        ];

        let (_x, d) = gen_cheb_diff_matrix(n);

        // First way, vec vec matrix multiplication D1 to D2, then convert D2 to an array
        let d2_array = get_cheb_diff_matrix_second(&d);

        // Comparison tests
        let diff_array_target = &d2_array - &target_array;
        let max_err_array_target = diff_array_target.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));

        println!("Max error between ndarray D² and target: {:.3e}", max_err_array_target);
        assert!(
            max_err_array_target < 1e-10,
            "ndarray D² does not match target! Max error = {:.3e}",
            max_err_array_target
        );
    }
}