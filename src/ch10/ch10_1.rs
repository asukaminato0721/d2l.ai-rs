#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Shape, Tensor as torch, Tensor, Var, D};
    use candle_nn::{
        self as nn, conv2d, conv2d_no_bias, linear, loss,
        ops::{self, sigmoid},
        seq, Activation, Conv2dConfig, Linear, Module, Optimizer, Sequential, VarBuilder, VarMap,
        RNN, SGD,
    };
    use rand::{seq::SliceRandom, thread_rng};

    #[test]
    fn ch_10_1_2_1() {
        struct LSTMScratch {
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
                let triple = || -> Result<(
                    Result<Tensor, Box<dyn std::error::Error>>,
                     Result<Tensor, Box<dyn std::error::Error>>,
                     Tensor), Box<dyn std::error::Error>> {
                        Ok((
                        init_weight((num_inputs, num_hiddens)),
                        init_weight((num_hiddens, num_hiddens)),
                        torch::zeros((num_hiddens), DType::F64, &Device::Cpu)?))

                };
                let (W_xi, W_hi, b_i) = triple()?; // Input gate
                let (W_xf, W_hf, b_f) = triple()?; // Forget gate
                let (W_xo, W_ho, b_o) = triple()?; // Output gate
                let (W_xc, W_hc, b_c) = triple()?; // Input node
                Ok(Self {
                    num_hiddens,
                    W_xi: W_xi?,
                    W_hi: W_hi?,
                    b_i,
                    W_xf: W_xf?,
                    W_hf: W_hf?,
                    b_f,
                    W_xo: W_xo?,
                    W_ho: W_ho?,
                    b_o,
                    W_xc: W_xc?,
                    W_hc: W_hc?,
                    b_c,
                })
            }

            fn forward(
                self,
                inputs: Vec<Tensor>,
                H_C: Option<(Tensor, Tensor)>,
            ) -> Result<(Vec<Tensor>, (Tensor, Tensor)), Box<dyn std::error::Error>> {
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
                            .add(&H.matmul(&self.W_hi)?)?
                            .add(&self.b_i)?,
                    )?;
                    let F = sigmoid(
                        &X.matmul(&self.W_xf)?
                            .add(&H.matmul(&self.W_hf)?)?
                            .add(&self.b_f)?,
                    )?;
                    let O = sigmoid(
                        &X.matmul(&self.W_xo)?
                            .add(&H.matmul(&self.W_ho)?)?
                            .add(&self.b_o)?,
                    )?;
                    let C_tilde = X
                        .matmul(&self.W_xc)?
                        .add(&H.matmul(&self.W_hc)?)?
                        .add(&self.b_c)?
                        .tanh()?;
                    let C = F.add(&C)?.add(&I.mul(&C_tilde)?)?;
                    let H = O.mul(&C.tanh()?)?;
                    outputs.push(H);
                }
                return Ok((outputs, (H, C)));
            }
        }
    }

    #[test]
    fn ch_10_1_3() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

        let lstm = nn::lstm(11, 111, Default::default(), vb.pp("lstm"))?;
        Ok(())
    }
}
