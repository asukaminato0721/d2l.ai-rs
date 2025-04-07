#[cfg(test)]
mod test {
    use candle_core::{DType, Device, Result, Tensor, Tensor as torch};
    use candle_nn::{
        Activation, Activation as nn, Conv2d, Conv2dConfig, Linear, Module, Sequential, VarBuilder,
        VarMap, conv2d, linear, ops::dropout, seq,
    };

    #[test]
    fn f() -> Result<()> {
        struct Inception {
            b1_1: Conv2d,
            b2_1: Conv2d,
            b2_2: Conv2d,
            b3_1: Conv2d,
            b3_2: Conv2d,
            //   b4_1: Conv2d,
            b4_2: Conv2d,
        }
        impl Inception {
            fn new(
                c1: usize,
                c2: (usize, usize),
                c3: (usize, usize),
                c4: usize,
                vb: &VarBuilder,
            ) -> Result<Self> {
                Ok(Self {
                    b1_1: conv2d(42, c1, 1, Default::default(), vb.pp("11"))?,
                    b2_1: conv2d(42, c2.0, 1, Default::default(), vb.pp("11"))?,
                    b2_2: conv2d(
                        42,
                        c2.1,
                        3,
                        Conv2dConfig {
                            padding: 1,
                            ..Default::default()
                        },
                        vb.pp("11"),
                    )?,
                    b3_1: conv2d(42, c3.0, 1, Default::default(), vb.pp("11"))?,
                    b3_2: conv2d(
                        42,
                        c3.1,
                        5,
                        Conv2dConfig {
                            padding: 2,
                            ..Default::default()
                        },
                        vb.pp("11"),
                    )?,
                    // b4_1: |x:&Tensor|x.max_pool2d_with_stride(3, 1)?,
                    b4_2: conv2d(42, c4, 1, Default::default(), vb.pp("11"))?,
                })
            }
        }

        impl Module for Inception {
            fn forward(&self, xs: &Tensor) -> Result<Tensor> {
                let b1 = self.b1_1.forward(xs)?.relu()?;
                let b2 = self.b2_2.forward(&self.b2_1.forward(xs)?.relu()?)?.relu()?;
                let b3 = self.b3_2.forward(&self.b3_1.forward(xs)?.relu()?)?.relu()?;
                let b4 = self
                    .b4_2
                    // TODO nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    .forward(&xs.max_pool2d_with_stride(3, 1)?)?
                    .relu()?;
                torch::cat(&[b1, b2, b3, b4], 1)
            }
        }

        struct GoogleNet {
            net: Sequential,
        }
        impl GoogleNet {
            fn b1(vb: &VarBuilder) -> Result<Sequential> {
                Ok(seq()
                    .add(conv2d(
                        42,
                        64,
                        7,
                        Conv2dConfig {
                            padding: 3,
                            stride: 2,
                            ..Default::default()
                        },
                        vb.pp("1"),
                    )?)
                    .add(nn::Relu)
                    // TODO nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    .add_fn(|x| x.max_pool2d_with_stride(3, 2)))
            }
            fn b2(vb: &VarBuilder) -> Result<Sequential> {
                Ok(seq()
                    .add(conv2d(42, 64, 1, Default::default(), vb.pp("1"))?)
                    .add(nn::Relu)
                    .add(conv2d(
                        64,
                        192,
                        3,
                        Conv2dConfig {
                            padding: 1,
                            ..Default::default()
                        },
                        vb.pp("1"),
                    )?)
                    .add(nn::Relu)
                    // TODO nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    .add_fn(|x| x.max_pool2d_with_stride(3, 2)))
            }
            fn b3(vb: &VarBuilder) -> Result<Sequential> {
                Ok(seq()
                    .add(Inception::new(64, (96, 128), (16, 32), 32, vb)?)
                    .add(Inception::new(128, (128, 192), (32, 96), 64, vb)?)
                    .add_fn(|x| todo!("nn.MaxPool2d(kernel_size=3, stride=1, padding=1)"))
                    .add_fn(|x| x.max_pool2d_with_stride(3, 2)))
            }

            fn b4(vb: &VarBuilder) -> Result<Sequential> {
                Ok(seq()
                    .add(Inception::new(192, (96, 208), (16, 48), 64, vb)?)
                    .add(Inception::new(160, (112, 224), (24, 64), 64, vb)?)
                    .add(Inception::new(128, (128, 256), (24, 64), 64, vb)?)
                    .add(Inception::new(112, (144, 288), (32, 64), 64, vb)?)
                    .add(Inception::new(256, (160, 320), (32, 128), 128, vb)?)
                    .add_fn(|x| todo!("nn.MaxPool2d(kernel_size=3, stride=1, padding=1)"))
                    .add_fn(|x| x.max_pool2d_with_stride(3, 2)))
            }
            fn b5(vb: &VarBuilder) -> Result<Sequential> {
                Ok(seq()
                    .add(Inception::new(256, (160, 320), (32, 128), 128, vb)?)
                    .add(Inception::new(384, (192, 384), (48, 128), 128, vb)?)
                    .add_fn(|x| todo!("nn.MaxPool2d(kernel_size=3, stride=1, padding=1)"))
                    .add_fn(|x| x.avg_pool2d((1, 1)))
                    .add_fn(|x| x.flatten_all()))
            }

            fn new(lr: f32, num_classes: usize, vb: VarBuilder) -> Result<Self> {
                Ok(Self {
                    net: seq()
                        .add(GoogleNet::b1(&vb)?)
                        .add(GoogleNet::b2(&vb)?)
                        .add(GoogleNet::b3(&vb)?)
                        .add(GoogleNet::b4(&vb)?)
                        .add(GoogleNet::b5(&vb)?)
                        .add(linear(42, num_classes, vb)?),
                })
            }
        }
        let device = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);
        GoogleNet::new(0.1, 10, vb);
        Ok(())
        // TODO training
    }
}
