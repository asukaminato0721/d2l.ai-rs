use candle_core::{D, Tensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor as torch, Var};
struct SyntheticRegressionData {
    X: Tensor,
    y: Tensor,
    num_train: usize,
    batch_size: usize,
    num_val: usize,
}

impl SyntheticRegressionData {
    fn new(
        w: Tensor,
        b: f64,
        noise: f64,
        num_train: usize,
        num_val: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let device = &Device::cuda_if_available(0)?;
        let n = num_train + num_val;
        let lenw = w.dim(0)?;
        let X = torch::rand(0., 1., (n, lenw), device)?;
        let noise = torch::rand(0., noise, (n, 1), device)?;
        let y = X
            .matmul(&w.reshape(((), 1))?)?
            .affine(1., b)?
            .broadcast_add(&noise)?;
        Ok(Self {
            X,
            y,
            num_train,
            batch_size,
            num_val,
        })
    }
}

#[cfg(test)]
mod test {
    use candle_core::{Device, IndexOp, Result, Tensor};

    use super::SyntheticRegressionData;

    #[test]
    fn f() -> Result<()> {
        let device = &Device::cuda_if_available(0)?;
        let data = SyntheticRegressionData::new(
            Tensor::new(&[2., -3.4], device)?,
            4.2,
            0.01,
            1000,
            1000,
            32,
        )?;
        dbg!(data.X.i(0)?);
        dbg!(data.y.i(0)?);
        Ok(())
    }
}
