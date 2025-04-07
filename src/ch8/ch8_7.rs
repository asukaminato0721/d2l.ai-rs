#[cfg(test)]
mod test {
    use candle_core::{Device, Tensor as torch};
    use candle_nn::{
        batch_norm, conv2d, seq, Activation, BatchNormConfig, Conv2dConfig, Module, Sequential,
        VarBuilder, VarMap,
    };

    #[test]
    fn f() -> Result<(), Box<dyn std::error::Error>> {
        fn conv_block(num_channels: usize, vb: &VarBuilder) -> candle_core::Result<Sequential> {
            Ok(seq()
                //   .add(batch_norm(42, BatchNormConfig::default(), vb)?)
                .add(Activation::Relu)
                .add(conv2d(
                    42,
                    num_channels,
                    3,
                    Conv2dConfig {
                        padding: 1,
                        ..Default::default()
                    },
                    vb.pp("1"),
                )?))
        }

        struct DenseBlock {
            net: Sequential,
        }
        impl DenseBlock {
            fn new(
                num_convs: usize,
                num_channels: usize,
                vb: VarBuilder,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                let mut layer = seq();
                for _ in 0..num_convs {
                    layer = layer.add(conv_block(num_channels, &vb)?)
                }
                Ok(Self { net: layer })
            }
        }
        impl Module for DenseBlock {
            fn forward(
                &self,
                xs: &candle_core::Tensor,
            ) -> candle_core::Result<candle_core::Tensor> {
                torch::cat(&self.net.forward_all(xs)?, 1)
            }
        }
        let dev = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F64, dev);

        let blk = DenseBlock::new(2, 10, vb)?;
        let X = torch::randn(0., 1., (4, 3, 8, 8), dev)?;
        // TODO fixme
        // let Y = blk.forward(&X)?;
        // println!("{:?}", Y.shape());
        Ok(())
    }
}
