#[cfg(test)]
mod test {
    use candle_core::{Device, Tensor, Tensor as torch};
    use candle_nn::{
        batch_norm, conv2d, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Module, ModuleT,
        VarBuilder, VarMap,
    };

    #[test]
    fn f() -> Result<(), Box<dyn std::error::Error>> {
        struct Residual {
            conv1: Conv2d,
            conv2: Conv2d,
            conv3: Option<Conv2d>,
            bn1: BatchNorm,
            bn2: BatchNorm,
        }
        impl Residual {
            fn new(
                num_channels: usize,
                use_1x1conv: bool,
                strides: usize,
                vb: VarBuilder,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                Ok(Self {
                    conv1: conv2d(
                        6,
                        num_channels,
                        3,
                        Conv2dConfig {
                            padding: 1,
                            stride: strides,
                            ..Default::default()
                        },
                        vb.pp("c1"),
                    )?,
                    conv2: conv2d(
                        6,
                        num_channels,
                        3,
                        Conv2dConfig {
                            padding: 1,
                            ..Default::default()
                        },
                        vb.pp("c2"),
                    )?,
                    conv3: if use_1x1conv {
                        Some(conv2d(
                            6,
                            num_channels,
                            3,
                            Conv2dConfig {
                                stride: strides,
                                ..Default::default()
                            },
                            vb.pp("c3"),
                        )?)
                    } else {
                        None
                    },
                    bn1: batch_norm(6, BatchNormConfig::default(), vb.pp("bn1"))?,
                    bn2: batch_norm(6, BatchNormConfig::default(), vb.pp("bn2"))?,
                })
            }
        }
        impl Module for Residual {
            fn forward(
                &self,
                xs: &candle_core::Tensor,
            ) -> candle_core::Result<candle_core::Tensor> {
                let mut Y = self.conv1.forward(xs)?;

                Y = self.bn1.forward_t(&Y, false)?.relu()?;
                let mut X = xs.clone();
                if self.conv3.is_some() {
                    X = self.conv3.clone().unwrap().forward(&X)?;
                }
                Y.add(&X)?.relu()
            }
        }
        // TODO training
        let dev = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F64, dev);
        let blk = Residual::new(6, true, 2, vb)?;
        let X = torch::randn(0., 1., (4, 3, 6, 6), dev)?;
        // TODO  Fixme
        // println!("{:?}", blk.forward(&X)?.dims());
        Ok(())
    }
}
