#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Tensor as torch, Tensor, Var};
    use candle_nn::{
        self as nn, linear, seq, Activation, Linear, Module, Sequential, VarBuilder, VarMap,
    };
    #[test]
    fn get_start() -> Result<(), Box<dyn std::error::Error>> {
        // copy from https://github.com/huggingface/candle/issues/1065
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let net = nn::seq()
            .add(linear(20, 256, vb.pp("fc1"))?)
            .add(Activation::Relu)
            .add(linear(256, 10, vb.pp("fc2"))?);
        let X = torch::rand(0., 1., (2, 20), device)?;
        println!("{:?}", net.forward(&X)?.shape());
        Ok(())
    }
    #[test]
    fn ch6_1_134() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

        struct MLP {
            hidden: Linear,
            out: Linear,
        }
        impl MLP {
            fn new() -> Result<Self, Box<dyn std::error::Error>> {
                let device = &Device::Cpu;
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

                Ok(Self {
                    hidden: linear(20, 256, vb.pp("fc1"))?,
                    out: linear(256, 10, vb.pp("fc2"))?,
                })
            }
            fn forward(self, X: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
                Ok(self.out.forward(&self.hidden.forward(X)?.relu()?)?)
            }
        }
        let X = torch::rand(0., 1., (2, 20), device)?;
        let net = MLP::new()?;
        println!("{:?}", net.forward(&X)?.shape());
        // fn ch6_1_3() -> Result<(), Box<dyn std::error::Error>> {

        struct FixedHiddenMLP {
            rand_weight: Tensor,
            linear: Linear,
        }
        impl FixedHiddenMLP {
            fn new() -> Result<Self, Box<dyn std::error::Error>> {
                let device = &Device::Cpu;
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
            fn forward(&self, X: &Tensor) -> Result<Tensor, candle_core::Error> {
                let mut X = self.linear.forward(X)?;
                X = X.matmul(&self.rand_weight)?.affine(0., 1.)?.relu()?;
                // Reuse the fully connected layer. This is equivalent to sharing
                // parameters with two fully connected layers
                X = self.linear.forward(&X)?;
                // Control flow
                while X.abs()?.sum_all()?.to_scalar::<f64>()? > 1. {
                    X = X.affine(0.5, 0.)?;
                }
                return Ok(X.sum_all()?);
            }
        }
        let net = FixedHiddenMLP::new()?;
        println!("{}", net.forward(&X)?);

        struct NestMLP {
            net: Sequential,
            linear: Linear,
        }
        impl NestMLP {
            fn new() -> Result<Self, Box<dyn std::error::Error>> {
                let device = &Device::Cpu;
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
            fn forward(&self, X: &Tensor) -> Result<Tensor, candle_core::Error> {
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
