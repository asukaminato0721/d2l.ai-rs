#[cfg(test)]
mod test {
    use candle_core::Result;
    use candle_nn::{
        batch_norm, conv2d, linear, seq, Activation, BatchNorm, Sequential, VarBuilder,
    };

    #[test]
    fn f() {
        struct BNLeNet {
            net: Sequential,
        }
        impl BNLeNet {
            fn new(lr: f32, num_classes: usize, vb: VarBuilder) -> Result<Self> {
                Ok(Self {
                    net: seq()
                        // nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                        .add(conv2d(42, 6, 5, Default::default(), vb.pp("11"))?)
                        // .add(batch_norm(10, Default::default(), vb.pp("1"))?),
                        // nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                        .add(Activation::Sigmoid)
                        .add_fn(|x| x.avg_pool2d_with_stride(2, 2))
                        //
                        // nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                        .add_fn(|x| x.avg_pool2d(5))
                        //.add(BatchNorm)
                        //
                        // nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                        .add(Activation::Sigmoid)
                        .add_fn(|x| x.avg_pool2d_with_stride(2, 2))
                        //
                        // nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                        .add_fn(|x| x.flatten_all())
                        .add(linear(42, 128, vb.pp("11111"))?)
                        //.add(BatchNorm)
                        //
                        // nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                        .add(Activation::Sigmoid)
                        .add(linear(42, 84, vb.pp("11111"))?)
                        //.add(BatchNorm)
                        //
                        // nn.Sigmoid(), nn.LazyLinear(num_classes)
                        .add(Activation::Sigmoid)
                        .add(linear(42, num_classes, vb.pp("11111"))?),
                })
            }
        }
        // TODO training
    }
}
