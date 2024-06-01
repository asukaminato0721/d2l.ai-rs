#[cfg(test)]
mod test {
    use std::{borrow::BorrowMut, vec};

    use candle_core::{DType, Device, IndexOp, Tensor as torch, Tensor, Var};
    use candle_nn::{
        self as nn, linear, seq, Activation, Conv2dConfig, Linear, Module, Sequential, VarBuilder,
        VarMap,
    };

    #[test]
    fn ch7_2_3() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

        let device = &Device::Cpu;

        let X = torch::new(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], device)?;
        let K = torch::new(&[[0., 1.], [2., 3.]], device)?;
        fn corr2d(X: &Tensor, K: &Tensor) -> candle_core::Result<Tensor> {
            let device = &Device::Cpu;
            let (h, w) = (K.dim(0)?, K.dim(1)?);

            let mut Y = vec![vec![Default::default(); X.dim(1)? - w + 1]; X.dim(0)? - h + 1];
            for i in 0..Y.len() {
                for j in 0..Y[0].len() {
                    Y[i][j] = X
                        .i((i..(i + h), j..(j + w)))?
                        .mul(K)?
                        .sum_all()?
                        .to_scalar::<f64>()?
                }
            }
            Ok(torch::new(Y, device)?)
        }
        println!("corr2d(X, K) {}", corr2d(&X, &K)?);

        {
            struct Conv2D {
                weight: Tensor,
                bias: Tensor,
            }
            impl Module for Conv2D {
                fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
                    Ok(corr2d(xs, &self.weight)?.add(&self.bias)?)
                }
            }

            let mut X = torch::ones((6, 8), DType::F64, device)?;
            // https://github.com/huggingface/candle/issues/1441 lol
            X = torch::cat(
                &[
                    X.i((.., ..2))?,
                    torch::zeros((6, 4), DType::F64, device)?,
                    X.i((.., 6..))?,
                ],
                1,
            )?;
            println!("{X}");

            let K = torch::new(&[[1.0, -1.0]], device)?;

            let Y = corr2d(&X, &K)?;

            println!("{}", Y);
            println!("{}", corr2d(&X.t()?, &K)?);
            // TODO d2l said that conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False) , but candle only support usize, so skip
            // but Op has a https://docs.rs/candle-core/0.5.1/candle_core/op/enum.Op.html#variant.Conv2D
            let conv2d = nn::conv2d(1, 1, 2, Default::default(), vb.pp("conv1"))?;
            let X = X.reshape((1, 1, 6, 8))?;
            let Y = Y.reshape((1, 1, 6, 7))?;
        }
        let lr = 3e-2; // Learning rate

        // for i in 0..10 {
        //     let Y_hat = conv2d.forward(&X)?;
        //     let l = Y_hat.broadcast_sub(&Y)?.powf(2.)?;
        //     // Update the kernel
        //     l.sum_all()?
        //         .backward()?
        //         .get(conv2d.weight())
        //         .unwrap()
        //         .affine(lr, 0.)?;
        //     if (i + 1) % 2 == 0 {
        //         println!("epoch {}, loss {}", i + 1, l.sum_all()?);
        //     }
        // }
        Ok(())
    }
}
