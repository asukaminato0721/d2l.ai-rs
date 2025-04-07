#[cfg(test)]
mod test {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        Activation, Conv2dConfig, Sequential, VarBuilder, VarMap, conv2d, linear, linear_no_bias,
        ops::dropout, seq,
    };

    #[test]
    fn f() {
        struct AlexNet {
            net: Sequential,
        }
        impl AlexNet {
            fn new(lr: f32, num_classes: usize) -> Result<Self, Box<dyn std::error::Error>> {
                let dev = &Device::cuda_if_available(0)?;
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F64, dev);

                Ok(Self {
                    net: seq()
                        .add(conv2d(
                            42,
                            96,
                            11,
                            Conv2dConfig {
                                padding: 1,
                                stride: 4,
                                ..Default::default()
                            },
                            vb.pp("c1"),
                        )?)
                        //
                        .add(Activation::Relu)
                        .add_fn(|x| x.max_pool2d_with_stride(3, 2))
                        //
                        .add(conv2d(
                            96,
                            256,
                            5,
                            Conv2dConfig {
                                padding: 2,
                                ..Default::default()
                            },
                            vb.pp("c2"),
                        )?)
                        .add(Activation::Relu)
                        //
                        .add_fn(|x| x.max_pool2d_with_stride(3, 2))
                        //
                        .add(conv2d(
                            256,
                            384,
                            3,
                            Conv2dConfig {
                                padding: 1,
                                ..Default::default()
                            },
                            vb.pp("c3"),
                        )?)
                        .add(Activation::Relu)
                        //
                        .add(conv2d(
                            384,
                            384,
                            3,
                            Conv2dConfig {
                                padding: 1,
                                ..Default::default()
                            },
                            vb.pp("c4"),
                        )?)
                        .add(Activation::Relu)
                        //
                        .add(conv2d(
                            384,
                            256,
                            3,
                            Conv2dConfig {
                                padding: 1,
                                ..Default::default()
                            },
                            vb.pp("c5"),
                        )?)
                        .add(Activation::Relu)
                        //
                        .add_fn(|x| x.max_pool2d_with_stride(3, 2))
                        .add_fn(|x| x.flatten_all())
                        //
                        .add(linear(6400, 4096, vb.pp("l1"))?)
                        .add(Activation::Relu)
                        .add_fn(|xs| dropout(xs, 0.5))
                        //
                        .add(linear(4096, 4096, vb.pp("l1"))?)
                        .add(Activation::Relu)
                        .add_fn(|xs| dropout(xs, 0.5))
                        //
                        .add(linear(4096, num_classes, vb.pp("l2"))?),
                })
            }
        }
        // TODO training
    }
}
