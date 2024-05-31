#[cfg(test)]
mod test {
    use std::time;

    use candle_core::{DType, Device, IndexOp, Tensor as torch, Tensor, Var};
    #[test]
    fn get_start() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let n = 10000;
        let a = torch::ones(10000, DType::F32, device)?;
        let b = torch::ones(10000, DType::F32, device)?;
        let c = torch::zeros(10000, DType::F32, device)?;

        // nothing to do here

        Ok(())
    }
}
