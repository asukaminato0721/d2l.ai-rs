use candle_core::{DType, Device, IndexOp, Tensor as torch, Var};
use candle_core::{Tensor, D};
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
    ) -> Self {
        let device = &Device::Cpu;
        let n = num_train + num_val;
        let lenw = w.shape().dims1().unwrap();
        let X = torch::rand(0., 1., (n, lenw), device).unwrap();
        let noise = torch::rand(0., noise, (n, 1), device).unwrap();
        let y = X
            .matmul(&w.reshape(((), 1)).unwrap())
            .unwrap()
            .affine(0., b)
            .unwrap()
            .broadcast_add(&noise)
            .unwrap();
        Self {
            X,
            y,
            num_train,
            batch_size,
            num_val,
        }
    }
}

#[cfg(test)]
mod test {
    use candle_core::{Device, IndexOp, Tensor};

    use super::SyntheticRegressionData;

    #[test]
    fn f() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let data = SyntheticRegressionData::new(
            Tensor::new(&[2., -3.4], device)?,
            4.2,
            0.01,
            1000,
            1000,
            32,
        );
        dbg!(data.X.i(0)?);
        dbg!(data.y.i(0)?);
        Ok(())
    }
}
