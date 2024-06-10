/// TODO https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/vgg.rs
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
        struct VGG {
            net: Sequential,
            lr: f32,
        }
        impl VGG {
            fn new(
                arch: &[(usize, usize)],
                lr: f32,
                num_classes: usize,
                vb: VarBuilder,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                fn vgg_block(
                    num_convs: usize,
                    out_channels: usize,
                    vb: &VarBuilder,
                ) -> Result<Sequential, Box<dyn std::error::Error>> {
                    let mut layers = seq();
                    for _ in 0..(num_convs) {
                        layers = layers
                            .add(conv2d(
                                42, // Fixme
                                out_channels,
                                3,
                                Conv2dConfig {
                                    padding: 1,
                                    ..Default::default()
                                },
                                vb.pp(num_convs.to_string()),
                            )?)
                            .add(Activation::Relu);
                    }
                    layers = layers.add_fn(|x| x.max_pool2d_with_stride(2, 2));
                    return Ok(layers);
                }

                let mut layer = seq();
                for (num_convs, out_channels) in arch {
                    layer = layer.add(vgg_block(*num_convs, *out_channels, &vb)?)
                }

                Ok(Self {
                    net: layer
                        .add_fn(|x| x.flatten_all())
                        //
                        .add(linear(4096, 4096, vb.pp("l1"))?)
                        .add(Activation::Relu)
                        .add_fn(|xs| dropout(xs, 0.5))
                        //
                        .add(linear(4096, 4096, vb.pp("l1"))?)
                        .add(Activation::Relu)
                        .add_fn(|xs| dropout(xs, 0.5))
                        //
                        .add(linear(4096, num_classes, vb.pp("l2"))?),
                    lr,
                })
            }
        }
        // TODO training
        let vgg = VGG::new(&[(1, 1)], 1e-3, 10, vb);
        Ok(())
    }
}
