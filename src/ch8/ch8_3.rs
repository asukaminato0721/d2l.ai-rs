#[cfg(test)]
mod test {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        conv2d, linear, linear_no_bias, ops::dropout, seq, Activation, Conv2dConfig, Sequential,
        VarBuilder, VarMap,
    };

    #[test]
    fn f() -> Result<(), Box<dyn std::error::Error>> {
        let dev = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, dev);

        struct NiN {
            net: Sequential,
            lr: f32,
        }
        impl NiN {
            fn new(
                lr: f32,
                num_classes: usize,
                vb: VarBuilder,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                fn nin_block(
                    out_channels: usize,
                    kernel_size: usize,
                    strides: usize,
                    padding: usize,
                    vb: &VarBuilder,
                ) -> candle_core::Result<Sequential> {
                    Ok(seq()
                        .add(conv2d(
                            10, // TODO
                            out_channels,
                            kernel_size,
                            Conv2dConfig {
                                padding,
                                stride: strides,
                                ..Default::default()
                            },
                            vb.pp("1"),
                        )?)
                        .add(Activation::Relu)
                        .add(conv2d(10, out_channels, 1, Default::default(), vb.pp("1"))?)
                        .add(Activation::Relu)
                        .add(conv2d(10, out_channels, 1, Default::default(), vb.pp("1"))?)
                        .add(Activation::Relu))
                }

                Ok(Self {
                    net: seq()
                        .add(nin_block(96, 11, 4, 0, &vb)?)
                        .add_fn(|x| x.max_pool2d_with_stride(3, 2))
                        //
                        .add(nin_block(256, 5, 1, 2, &vb)?)
                        .add_fn(|x| x.max_pool2d_with_stride(3, 2))
                        //
                        .add(nin_block(384, 3, 1, 1, &vb)?)
                        .add_fn(|x| x.max_pool2d_with_stride(3, 2))
                        //
                        .add_fn(|xs| dropout(xs, 0.5))
                        .add(nin_block(num_classes, 3, 1, 1, &vb)?)
                        .add_fn(|x| x.avg_pool2d((1, 1)))
                        .add_fn(|x| x.flatten_all()),
                    lr,
                })
            }
        }
        let nin = NiN::new(1e-3, 10, vb);
        // TODO training
        Ok(())
    }
}
