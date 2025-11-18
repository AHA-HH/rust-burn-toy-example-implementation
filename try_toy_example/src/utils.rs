use burn::tensor::Tensor;    
use burn::tensor::backend::Backend;

use crate::cheb_points::gen_cheb_points;

// Function to use tensor Chebyshev points as tensorgrid
pub fn cheb_tensorgrid<B: Backend>(
    device: &B::Device,
    nx: usize,
    ny: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let x = gen_cheb_points::<B>(device, nx);
    let y = gen_cheb_points::<B>(device, ny);

    // X, Y each shape [ny, nx]
    let x_grid = x.clone().unsqueeze_dim(0).repeat(&[ny, 1]);        // replicate x across rows
    let y_grid = y.clone().unsqueeze_dim(1).repeat(&[1, nx]);        // replicate y down columns

    (x_grid, y_grid)
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use super::cheb_tensorgrid;

    type B = NdArray<f32>;

    #[test]
    fn test_cheb_tensorgrid() {
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let (nx, ny) = (5, 4);

        let (x_grid, y_grid) = cheb_tensorgrid::<B>(&device, nx, ny);

        // Expected shapes
        assert_eq!(x_grid.dims(), [ny, nx], "X grid shape should be [ny, nx]");
        assert_eq!(y_grid.dims(), [ny, nx], "Y grid shape should be [ny, nx]");

        let x_data = x_grid.to_data().convert::<f32>().to_vec().expect("Failed to convert tensor to Vec");
        let y_data = y_grid.to_data().convert::<f32>().to_vec().expect("Failed to convert tensor to Vec");

        // Check corner values
        let x_first_row: Vec<f32> = x_data[0..nx].to_vec();
        let y_first_col: Vec<f32> = (0..ny).map(|i| y_data[i * nx]).collect();

        println!("X first row: {:?}", x_first_row);
        println!("Y first column: {:?}", y_first_col);

        // x repeats across rows, y repeats down columns
        for i in 1..ny {
            let row_start = i * nx;
            let row_end = row_start + nx;
            assert_eq!(
                &x_data[0..nx],
                &x_data[row_start..row_end],
                "X grid does not repeat across rows"
            );
        }

        for j in 1..nx {
            for i in 0..ny {
                assert!(
                    (y_data[i * nx + j] - y_data[i * nx]).abs() < 1e-6,
                    "Y grid does not repeat down columns"
                );
            }
        }
    }
}