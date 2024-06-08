#[cfg(test)]
mod test {
    use std::{borrow::BorrowMut, vec};

    use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
    use candle_nn::{
        self as nn, conv, conv2d, conv2d_no_bias, linear, loss, ops, seq, Activation, Conv2dConfig,
        Linear, Module, Optimizer, Sequential, VarBuilder, VarMap,
    };
    use rand::{prelude::SliceRandom, thread_rng};

    #[test]
    fn cnn() -> Result<(), Box<dyn std::error::Error>> {
        struct LeNet {
            net: Sequential,
        }
        impl LeNet {
            fn new(num_classes: usize, vb: VarBuilder) -> Result<Self, Box<dyn std::error::Error>> {
                Ok(Self {
                    net: nn::seq()
                        // d2l said this, but I can't figure out the dims
                        // it passes the mnist, but not the fashion-mnist
                        //     .add(
                        //         conv2d(
                        //         64,
                        //         6,
                        //         5,
                        //         Conv2dConfig {
                        //             padding: 2,
                        //             ..Default::default()
                        //         },
                        //         vb.pp("conv"),
                        //     )?
                        // )
                        //  .add(Activation::Sigmoid)
                        //    .add_fn(|x| x.avg_pool2d_with_stride(2, 2))
                        //   .add(conv2d(6, 16, 5, Default::default(), vb.pp("conv1"))?)
                        //   .add(Activation::Sigmoid)
                        //    .add_fn(|x| x.avg_pool2d_with_stride(2, 2))
                        .add(linear(784, 120, vb.pp("fc1"))?)
                        .add(linear(120, 84, vb.pp("fc2"))?)
                        .add(linear(84, num_classes, vb.pp("fc3"))?),
                })
            }
            fn forward(&self, X: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
                Ok(self.net.forward(X)?)
            }
        }

        // mostly copy from https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs
        let BSIZE = 64;
        let dev = &Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = LeNet::new(11, vb)?;
        let m = candle_datasets::vision::mnist::load_dir("data")?;
        let train_labels = m.train_labels;
        let train_images = m.train_images.to_device(dev)?;
        let train_labels = train_labels.to_dtype(DType::U32)?.to_device(dev)?;
        // http://d2l.ai/_modules/d2l/torch.html#Trainer use SGD so I also use it.
        let mut opt = candle_nn::SGD::new(varmap.all_vars(), 0.1)?;
        let test_images = m.test_images.to_device(&dev)?;
        let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        let n_batches = train_images.dim(0)? / BSIZE;
        let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
        for epoch in 1..10 {
            let mut sum_loss = 0f32;
            batch_idxs.shuffle(&mut thread_rng());
            for batch_idx in batch_idxs.iter() {
                let train_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
                let train_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;
                let logits = model.forward(&train_images)?;
                let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &train_labels)?;
                // dbg!(&loss);
                opt.backward_step(&loss)?;
                sum_loss += loss.to_vec0::<f32>()?;
            }
            // dbg!(sum_loss);
            let avg_loss = sum_loss / n_batches as f32;

            let test_logits = model.forward(&test_images)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(&test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_vec0::<f32>()?;
            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
            println!(
                "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
                avg_loss,
                100. * test_accuracy
            );
        }
        Ok(())
    }
}
// cargo test --package d2l --lib -r -- ch7::ch7_6::test --nocapture
