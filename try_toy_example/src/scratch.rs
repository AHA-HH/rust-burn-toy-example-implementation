// pub fn train_model<B: Backend>(
//     device: &B::Device,
//     epochs: usize,
//     n: usize,
//     m: usize,
//     hidden_size: usize,
//     learning_rate: f32,
// ) {
//     type AD = Autodiff<B>;

//     let (x_cheb, d1) = gen_cheb_diff_matrix(n);
//     let d2 = get_cheb_diff_matrix_second(&d1);
//     let b_weights = gen_barycentric_weights(n);
//     let x_rand = gen_collocation_points::<AD>(device, m);
//     let cc_weights = gen_clenshaw_curtis_weights(m - 1);

//     let model: TwoLayerNet<AD> = TwoLayerNetConfig {
//         input_features: 1,
//         hidden_features: hidden_size,
//         output_features: 1,
//     }
//     .init(device);

//     let mut optim = AdamConfig::new()
//         .init();

//     // --- Training loop ---
//     for epoch in 0..epochs {
//         let x_tensor = Tensor::<AD, 2>::from_floats(x_cheb.as_slice(), device).reshape([n, 1]);
//         let u_pred = model.forward(x_tensor.clone());

//         let residuals = compute_residual::<AD>(
//             &u_pred,
//             x_cheb.clone(),
//             &d2,
//             &x_rand,
//             b_weights.clone(),
//             device,
//         );
//         let loss = compute_loss::<AD>(&residuals, &cc_weights, device);

    
//         optim.backward_step(&loss, &model);

//         let loss_val = loss.to_data().to_vec::<f32>().unwrap()[0];
//         println!("Epoch {:>3}/{:>3} | Loss: {:.6e}", epoch + 1, epochs, loss_val);
//     }
// }

