use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use scirs2_fft::{dct, idct, DCTType};

// Function to carry out the DCT of Type II in 1D, from values to coefficients
pub fn dct_1d<B: Backend>(
    input: &Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 1> {
    // Convert tensor input to a vector for crate usage
    let input_vec = input.to_data().to_vec::<f64>().unwrap();

    // Compute DCT of input vector
    let dct_coeffs_vec = dct(&input_vec, Some(DCTType::Type2), Some("ortho")).expect("Operation failed");

    // Convert back to a tensor
    Tensor::<B, 1>::from_data(&*dct_coeffs_vec, device)
}

// Function to carry out the inverse DCT of Type II in 1D, from coefficients to values
pub fn idct_1d<B: Backend>(
    input: &Tensor<B, 1>,
    device: & B::Device,
) -> Tensor<B, 1> {
    // Convert tensor input to a vector for crate usage
    let input_vec = input.to_data().to_vec::<f64>().unwrap();

    // Compute inverse DCT of input vector
    let idct_coeffs_vec = idct(&input_vec, Some(DCTType::Type2), Some("ortho")).expect("Operation failed");

    // Convert back to a tensor
    Tensor::<B, 1>::from_data(&*idct_coeffs_vec, device)
}

// Function to apply the spectral derivative

#[cfg(test)]
mod test {
    use burn::tensor::Tensor;
    use burn::backend::NdArray;
    use burn::tensor::backend::Backend;

    type B = NdArray<f64>;

    use crate::{
        cheb_points::gen_cheb_points, spectral::*
    };

    #[test]
    fn test_dct() {
        let device = <B as Backend>::Device::default();
        
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let signal_tensor = Tensor::<B, 1>::from_data(signal.as_slice(), &device);

        let dct_coeffs_tensor = dct_1d(&signal_tensor, &device);

        let dct_coeffs_vec = dct_coeffs_tensor.to_data().to_vec::<f64>().unwrap();

        let mean = 2.5;
        assert!((dct_coeffs_vec[0] / 2.0 - mean).abs() < 1e-10);
    }

    #[test]
    fn test_idct() {
        let device = <B as Backend>::Device::default();
        
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let signal_tensor = Tensor::<B, 1>::from_data(signal.as_slice(), &device);

        let dct_coeffs_tensor = dct_1d(&signal_tensor, &device);

        let recovered_tensor = idct_1d(&dct_coeffs_tensor, &device);

        let recovered = recovered_tensor.to_data().to_vec::<f64>().unwrap();

        for (i, &val) in signal.iter().enumerate() {
            assert!((val - recovered[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_cheb_points_fft() {
        let device = <B as Backend>::Device::default();

        let n = 5;
        let points = gen_cheb_points::<B>(&device, n);
        let points_vec = points.to_data().to_vec::<f64>().unwrap();

        let dct_coeffs = dct_1d(&points, &device);

        let recovered_tensor = idct_1d(&dct_coeffs, &device);

        let recovered = recovered_tensor.to_data().to_vec::<f64>().unwrap();

        for (i, &val) in points_vec.iter().enumerate() {
            assert!((val - recovered[i]).abs() < 1e-15);
        }
    }

}



