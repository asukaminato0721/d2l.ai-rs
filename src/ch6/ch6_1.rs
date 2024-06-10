#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Result, Tensor as torch, Tensor, Var};
    use candle_nn::{
        self as nn, linear, seq, Activation, Linear, Module, Sequential, VarBuilder, VarMap,
    };
    #[test]
    fn get_start() -> Result<()> {
        // copy from https://github.com/huggingface/candle/issues/1065
        let device = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);
        let net = nn::seq()
            .add(linear(20, 256, vb.pp("fc1"))?)
            .add(Activation::Relu)
            .add(linear(256, 10, vb.pp("fc2"))?);
        let X = torch::rand(0., 1., (2, 20), device)?;
        assert_eq!(net.forward(&X)?.dims(), [2, 10]);
        Ok(())
    }
    #[test]
    fn ch6_1_134() -> Result<()> {
        let device = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

        struct MLP {
            hidden: Linear,
            out: Linear,
        }
        impl MLP {
            fn new() -> Result<Self> {
                let device = &Device::cuda_if_available(0)?;
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

                Ok(Self {
                    hidden: linear(20, 256, vb.pp("fc1"))?,
                    out: linear(256, 10, vb.pp("fc2"))?,
                })
            }
            fn forward(self, X: &Tensor) -> Result<Tensor> {
                Ok(self.out.forward(&self.hidden.forward(X)?.relu()?)?)
            }
        }
        let X = torch::rand(0., 1., (2, 20), device)?;
        let net = MLP::new()?;
        assert_eq!(net.forward(&X)?.dims(), [2, 10]);

        struct FixedHiddenMLP {
            rand_weight: Tensor,
            linear: Linear,
        }
        impl FixedHiddenMLP {
            fn new() -> Result<Self> {
                let device = &Device::cuda_if_available(0)?;
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

                // Random weight parameters that will not compute gradients and
                // therefore keep constant during training
                Ok(Self {
                    rand_weight: torch::rand(0., 1., (20, 20), device)?,
                    linear: linear(20, 20, vb.pp("fc1"))?,
                })
            }
        }
        impl Module for FixedHiddenMLP {
            fn forward(&self, X: &Tensor) -> Result<Tensor> {
                let mut X = self.linear.forward(X)?;
                X = (X.matmul(&self.rand_weight)? + 1.)?.relu()?;
                // Reuse the fully connected layer. This is equivalent to sharing
                // parameters with two fully connected layers
                X = self.linear.forward(&X)?;
                // Control flow
                while X.abs()?.sum_all()?.to_vec0::<f64>()? > 1. {
                    X = (X * 0.5)?;
                }
                return X.sum_all();
            }
        }
        let net = FixedHiddenMLP::new()?;
        println!("{}", net.forward(&X)?);

        struct NestMLP {
            net: Sequential,
            linear: Linear,
        }
        impl NestMLP {
            fn new() -> Result<Self> {
                let device = &Device::cuda_if_available(0)?;
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

                Ok(Self {
                    net: seq()
                        .add(linear(20, 64, vb.pp("fc1"))?)
                        .add(Activation::Relu)
                        .add(linear(64, 32, vb.pp("fc2"))?)
                        .add(Activation::Relu),
                    linear: linear(32, 16, vb.pp("fc3"))?,
                })
            }
        }
        impl Module for NestMLP {
            fn forward(&self, X: &Tensor) -> Result<Tensor> {
                Ok(self.linear.forward(&self.net.forward(X)?)?)
            }
        }
        let chimera = seq()
            .add(NestMLP::new()?)
            .add(linear(16, 20, vb.pp("fcN"))?)
            .add(FixedHiddenMLP::new()?);
        println!("{}", chimera.forward(&X)?);

        Ok(())
    }
}
