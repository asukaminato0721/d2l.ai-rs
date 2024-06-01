#[cfg(test)]
mod test {
    use std::{borrow::BorrowMut, vec};

    use candle_core::{DType, Device, IndexOp, Tensor as torch, Tensor, Var};
    use candle_nn::{
        self as nn, conv2d, conv2d_no_bias, linear, seq, Activation, Conv2dConfig, Linear, Module,
        Sequential, VarBuilder, VarMap,
    };

    #[test]
    fn cnn() -> Result<(), Box<dyn std::error::Error>> {
        struct LeNet {
            net: Sequential,
        }
        impl LeNet {
            fn new(num_classes: usize) -> Result<Self, Box<dyn std::error::Error>> {
                let device = &Device::Cpu;
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

                Ok(Self {
                    net: nn::seq()
                        .add(conv2d(
                            2,
                            6,
                            5,
                            Conv2dConfig {
                                padding: 2,
                                ..Default::default()
                            },
                            vb.pp("conv"),
                        )?)
                        .add(Activation::Sigmoid)
                        .add_fn(|x| x.avg_pool2d_with_stride(2, 2))
                        .add(conv2d(6, 16, 5, Default::default(), vb.pp("conv1"))?)
                        .add(Activation::Sigmoid)
                        .add_fn(|x| x.avg_pool2d_with_stride(2, 2))
                        .add(linear(16, 120, vb.pp("fc1"))?)
                        .add(linear(120, 84, vb.pp("fc2"))?)
                        .add(linear(84, num_classes, vb.pp("fc3"))?),
                })
            }
        }
        // TODO training
        Ok(())
    }
}
