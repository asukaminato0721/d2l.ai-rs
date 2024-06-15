#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Shape, Tensor, Tensor as torch, Var};
    use candle_nn::{
        self as nn, conv2d, conv2d_no_bias, linear, ops::sigmoid, seq, Activation, Conv2dConfig,
        Linear, Module, Sequential, VarBuilder, VarMap,
    };

    #[test]
    fn ch_10_2_4() {
        struct GRUScratch {
            num_hiddens: usize,

            W_xz: Tensor,
            W_hz: Tensor,
            b_z: Tensor, // Input gate

            W_xr: Tensor,
            W_hr: Tensor,
            b_r: Tensor, // Forget gate

            W_xh: Tensor,
            W_hh: Tensor,
            b_h: Tensor, // Output gate
        }
        impl GRUScratch {
            fn new(
                num_inputs: usize,
                num_hiddens: usize,
                sigma: f64,
            ) -> Result<Self, candle_core::Error> {
                let init_weight = |shape: (usize, usize)| -> Result<Tensor, candle_core::Error> {
                    torch::rand(0., 1., shape, &Device::Cpu)? * sigma
                };
                let triple = || -> Result<(
                    Result<Tensor, candle_core::Error>,
                     Result<Tensor, candle_core::Error>,
                     Tensor), candle_core::Error> {
                        Ok((
                        init_weight((num_inputs, num_hiddens)),
                        init_weight((num_hiddens, num_hiddens)),
                        torch::zeros((num_hiddens), DType::F64, &Device::Cpu)?))

                };
                let (W_xz, W_hz, b_z) = triple()?; // Update gate
                let (W_xr, W_hr, b_r) = triple()?; // Reset gate
                let (W_xh, W_hh, b_h) = triple()?; // Candidate hidden state
                Ok(Self {
                    num_hiddens,
                    W_xz: W_xz?,
                    W_hz: W_hz?,
                    b_z,
                    W_xr: W_xr?,
                    W_hr: W_hr?,
                    b_r,
                    W_xh: W_xh?,
                    W_hh: W_hh?,
                    b_h,
                })
            }

            fn forward(
                self,
                inputs: Vec<Tensor>,
                H_C: Option<Tensor>,
            ) -> Result<(Vec<Tensor>, Tensor), Box<dyn std::error::Error>> {
                let H = if H_C.is_none() {
                    // Initial state with shape: (batch_size, num_hiddens)
                    torch::zeros(
                        (inputs[0].dim(0)?, self.num_hiddens),
                        DType::F64,
                        &Device::Cpu,
                    )?
                } else {
                    H_C.unwrap()
                };
                let mut outputs = vec![];
                for X in inputs {
                    let Z = sigmoid(
                        &X.matmul(&self.W_xz)?
                            .add(&H.matmul(&self.W_hz)?)?
                            .add(&self.b_z)?,
                    )?;
                    let R = sigmoid(
                        &X.matmul(&self.W_xr)?
                            .add(&H.matmul(&self.W_hr)?)?
                            .add(&self.b_r)?,
                    )?;
                    let H_tilde = X
                        .matmul(&self.W_xh)?
                        .add(&R.mul(&H)?.matmul(&self.W_hh)?)?
                        .add(&self.b_h)?
                        .tanh()?;

                    let H = Z.mul(&H)?.add(&Z.affine(-1., 1.)?.mul(&H_tilde)?)?;
                    outputs.push(H);
                }
                return Ok((outputs, H));
            }
        }
    }

    #[test]
    fn ch_10_2_5() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

        let gru = nn::gru(11, 111, Default::default(), vb.pp("gru"))?;
        Ok(())
    }
}
