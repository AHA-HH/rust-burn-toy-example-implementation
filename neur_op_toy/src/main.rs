// Rough translation of Timo Julia Chebychev code to Rust

fn main(){
    let n = 16;
    
    let points = cheb_points(n);
    
    for (j, x_j) in points {
        println!("Point {}: {}", j, x_j);
    }

    let weights = cheb_weights(n);
    for (j, w_j) in weights.iter().enumerate() {
        println!("Weight {}: {}", j, w_j);
    }
}


pub fn cheb_points(n: usize) -> Vec<(usize, f64)> {
    let pi = std::f64::consts::PI;
    let indices = (0..n).rev();
    
    let mut points: Vec<(usize, f64)> = Vec::with_capacity(n);
    
    for j in indices {
        let theta = (j as f64) * pi / ((n - 1) as f64);

        let x_j = theta.cos();

        points.push((j, x_j));
    }

    points
}

pub fn cheb_weights(n: usize) -> Vec<f64> {
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

pub fn cheb_1d_interpolate(eval_points, values, c_points, c_weights) {
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
            
            if diff.abs() <= 1e-14 * (1.0_f64.max(xeval.abs())) {
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

