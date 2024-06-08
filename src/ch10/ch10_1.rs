#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use candle_core::{DType, Device, IndexOp, Shape, Tensor as torch, Tensor, Var, D};
    use candle_nn::{
        self as nn, conv2d, conv2d_no_bias, linear, loss,
        ops::{self, sigmoid},
        seq, Activation, Conv2dConfig, LSTMConfig, Linear, Module, Optimizer, Sequential,
        VarBuilder, VarMap, RNN, SGD,
    };
    use rand::{seq::SliceRandom, thread_rng};
    struct LSTMScratch {
        num_inputs: usize,
        num_hiddens: usize,

        W_xi: Tensor,
        W_hi: Tensor,
        b_i: Tensor, // Input gate

        W_xf: Tensor,
        W_hf: Tensor,
        b_f: Tensor, // Forget gate

        W_xo: Tensor,
        W_ho: Tensor,
        b_o: Tensor, // Output gate

        W_xc: Tensor,
        W_hc: Tensor,
        b_c: Tensor, // Input node
    }
    impl LSTMScratch {
        fn new(
            num_inputs: usize,
            num_hiddens: usize,
            sigma: f64,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let init_weight =
                |shape: (usize, usize)| -> Result<Tensor, Box<dyn std::error::Error>> {
                    Ok(torch::rand(0., 1., shape, &Device::Cpu)?.affine(sigma, 0.)?)
                };
            let triple = || -> Result<(Tensor, Tensor, Tensor), Box<dyn std::error::Error>> {
                Ok((
                    init_weight((num_inputs, num_hiddens))?,
                    init_weight((num_hiddens, num_hiddens))?,
                    torch::zeros((num_hiddens), DType::F64, &Device::Cpu)?,
                ))
            };
            let (W_xi, W_hi, b_i) = triple()?; // Input gate
            let (W_xf, W_hf, b_f) = triple()?; // Forget gate
            let (W_xo, W_ho, b_o) = triple()?; // Output gate
            let (W_xc, W_hc, b_c) = triple()?; // Input node
            Ok(Self {
                num_inputs,
                num_hiddens,
                W_xi,
                W_hi,
                b_i,
                W_xf,
                W_hf,
                b_f,
                W_xo,
                W_ho,
                b_o,
                W_xc,
                W_hc,
                b_c,
            })
        }

        fn forward(
            &self,
            inputs: &[Tensor],
            H_C: Option<(Tensor, Tensor)>,
        ) -> Result<(Vec<Tensor>, Option<(Tensor, Tensor)>), Box<dyn std::error::Error>> {
            let (H, C) = if H_C.is_none() {
                // Initial state with shape: (batch_size, num_hiddens)
                (
                    torch::zeros(
                        (inputs[0].dim(0)?, self.num_hiddens),
                        DType::F64,
                        &Device::Cpu,
                    )?,
                    torch::zeros(
                        (inputs[0].dim(0)?, self.num_hiddens),
                        DType::F64,
                        &Device::Cpu,
                    )?,
                )
            } else {
                H_C.unwrap()
            };
            let mut outputs = vec![];
            for X in inputs {
                let I = sigmoid(
                    &X.matmul(&self.W_xi)?
                        .broadcast_add(&H.matmul(&self.W_hi)?)?
                        .broadcast_add(&self.b_i)?,
                )?;
                let F = sigmoid(
                    &X.matmul(&self.W_xf)?
                        .broadcast_add(&H.matmul(&self.W_hf)?)?
                        .broadcast_add(&self.b_f)?,
                )?;
                let O = sigmoid(
                    &X.matmul(&self.W_xo)?
                        .broadcast_add(&H.matmul(&self.W_ho)?)?
                        .broadcast_add(&self.b_o)?,
                )?;
                let C_tilde = X
                    .matmul(&self.W_xc)?
                    .broadcast_add(&H.matmul(&self.W_hc)?)?
                    .broadcast_add(&self.b_c)?
                    .tanh()?;
                let C = F.broadcast_add(&C)?.broadcast_add(&I.mul(&C_tilde)?)?;
                let H = O.mul(&C.tanh()?)?;
                outputs.push(H);
            }
            return Ok((outputs, Some((H, C))));
        }
    }
    #[test]
    fn ch_10_1_2_1() {}

    #[test]
    fn ch_10_1_3() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::cuda_if_available(0)?;
        let dev = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

        use crate::utils::TimeMachine;
        /// data = d2l.TimeMachine(batch_size=1024, num_steps=32)
        let data = TimeMachine::new(1024, 32, 10000, 5000)?;
        let lstm = LSTMScratch::new(data.vocab.len(), 32, 0.5)?;
        // idk wtf it is, but it seems to be worked. :P

        let mut outputs = vec![];
        let mut H_C = None;
        let mut output;

        for inp in [3f64, 1., 4., 1., 5., 9., 2.] {
            let inp = Tensor::new(&[[inp; 27]; 27], dev)?;
            (output, H_C) = lstm.forward(&[inp], H_C.clone())?;
            outputs.push(output);
        }
        println!("{:?}", outputs);
        Ok(())
    }
}
