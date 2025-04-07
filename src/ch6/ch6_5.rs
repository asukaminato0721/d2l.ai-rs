#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Tensor as torch, Tensor, Var};
    use candle_nn::{
        self as nn, linear, seq, Activation, Linear, Module, Sequential, VarBuilder, VarMap,
    };
    #[test]
    fn ch6_5_1() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

        let X = torch::rand(0., 1., (2, 20), device)?;

        struct CenteredLayer {}

        impl Module for CenteredLayer {
            fn forward(&self, X: &Tensor) -> Result<Tensor, candle_core::Error> {
                X.broadcast_sub(&X.mean_all()?)
            }
        }
        let net = CenteredLayer {};
        assert_eq!(
            net.forward(&Tensor::arange(1., 6., device)?)?
                .to_vec1::<f64>()?,
            [-2., -1., 0., 1., 2.]
        );
        let net = nn::seq()
            .add(linear(8, 128, vb.pp("fc1"))?)
            .add(CenteredLayer {});
        let Y = net.forward(&torch::rand(0., 1., (4, 8), device)?)?;
        println!("{}", Y.mean_all()?);

        struct MyLinear {
            weight: Tensor,
            bias: Tensor,
        }
        impl MyLinear {
            fn new(in_units: usize, units: usize) -> Result<Self, Box<dyn std::error::Error>> {
                let device = &Device::cuda_if_available(0)?;
                Ok(Self {
                    weight: torch::rand(0., 1., (in_units, units), device)?,
                    bias: torch::rand(0., 1., (units,), device)?,
                })
            }
        }
        impl Module for MyLinear {
            fn forward(&self, X: &Tensor) -> Result<Tensor, candle_core::Error> {
                X.matmul(&self.weight)?.broadcast_add(&self.bias)
            }
        }
        let linear = MyLinear::new(5, 3)?;
        println!("{}", linear.weight);
        println!(
            "{}",
            linear.forward(&Tensor::rand(0., 1., (2, 5), device)?)?
        );
        let net = nn::seq()
            .add(MyLinear::new(64, 8)?)
            .add(MyLinear::new(8, 1)?);
        println!("{}", net.forward(&Tensor::rand(0., 1., (2, 64), device)?)?);
        Ok(())
    }
}
