use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use scirs2_fft::{dct, idct, DCTType};
use sciml_rs::chebychev::Derivative1D;


// Function to carry out the DCT of Type II in 1D, from values to coefficients
pub fn dct_1d<B: Backend>(
    values: &Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 1> {
    // Convert tensor input to a vector for crate usage
    let values_vec = values.to_data().to_vec::<f64>().unwrap();

    // Compute DCT of input vector
    let coefficients_vec = dct(&values_vec, Some(DCTType::Type2), Some("ortho")).expect("Operation failed");

    // Convert back to a tensor
    Tensor::<B, 1>::from_data(&*coefficients_vec, device)

}

// Function to carry out the inverse DCT of Type II in 1D, from coefficients to values
pub fn idct_1d<B: Backend>(
    coefficients: &Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 1> {
    // Convert tensor input to a vector for crate usage
    let coefficients_vec = coefficients.to_data().to_vec::<f64>().unwrap();

    // Compute inverse DCT of input vector
    let values_vec = idct(&coefficients_vec, Some(DCTType::Type2), Some("ortho")).expect("Operation failed");

    // Convert back to a tensor
    Tensor::<B, 1>::from_data(&*values_vec, device)
}

// Function to carry out the recursion for the derivative of the Chebyshev series
pub fn cheb_derivative_recursion<B: Backend>(
    coefficients: &Tensor<B, 1>,
    device: &B::Device,
) -> Tensor<B, 1> {
    coefficients.clone().chebychev_derivative_1d()
}

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

        let n = 20;
        let points = gen_cheb_points::<B>(&device, n);
        let points_vec = points.to_data().to_vec::<f64>().unwrap();

        let dct_coeffs = dct_1d(&points, &device);

        let recovered_tensor = idct_1d(&dct_coeffs, &device);

        let recovered = recovered_tensor.to_data().to_vec::<f64>().unwrap();

        for (i, &val) in points_vec.iter().enumerate() {
            assert!((val - recovered[i]).abs() < 1e-14);
        }
    }

    // #[test]
    // fn test_sciml_derivative_import() {
    //     let device = <B as Backend>::Device::default();

    //     let test_coeffs = Tensor::<B, 1>::from_data([1, 1, 1].as_slice(), &device);

    //     let deriv_1_tensor = cheb_derivative_recursion(&test_coeffs, &device);

    //     let deriv_1 = deriv_1_tensor.to_data().to_vec::<f64>().unwrap();

    //     let expected = vec![1.0, 4.0, 0.0];

    //     for (i, &val) in expected.iter().enumerate() {
    //         assert!((val - deriv_1[i]).abs() < 1e-14);
    //     }        
    // }

    #[test]
    fn test_derivative_recursion() {
        let device = <B as Backend>::Device::default();

        let n = 5;
        let points = gen_cheb_points::<B>(&device, n);
        
        // Testing function sinx
        let values = points.clone().sin();

        // First derivative cosx
        let expected = points.clone().cos();
        let expected_deriv_1 = expected.to_data().to_vec::<f64>().unwrap();

        // Second derivative -sinx
        let expected_2 = values.clone().neg();
        // let expected_2: Vec<f64> = values.iter().map(|&x| -x).collect();

        // Compute coefficients
        let coeffs = dct_1d(&values.clone(), &device);

        // Calculate first derivative
        let deriv_1 = cheb_derivative_recursion(&coeffs, &device);

        // Compute values
        let calculated = idct_1d(&deriv_1, &device);

        let computed_deriv_1 = calculated.to_data().to_vec::<f64>().unwrap();

        // println!("Vector {:#?}", expected_deriv_1);

        // println!("Vector {:#?}", computed_deriv_1);

        // Compare first derivative
        for (i, &val) in expected_deriv_1.iter().enumerate() {
            assert!((val - computed_deriv_1[i]).abs() < 1e-14);
        }

        // Compute coefficients

    }

}



