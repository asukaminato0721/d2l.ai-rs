#[cfg(test)]
mod test {
    use std::time::{self, SystemTime};

    use candle_core::{DType, Device, IndexOp, Tensor as torch, Tensor, Var};
    #[test]
    fn get_start() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::cuda_if_available(0)?;
        let n = 10000;
        let a = torch::ones(n, DType::F32, device)?;
        let b = torch::ones(n, DType::F32, device)?;
        let mut c = vec![Default::default(); n];
        let t = time::SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_millis();
        for i in 0..n {
            c[i] = a.i(i)?.add(&b.i(i)?)?.to_vec0::<f32>()?;
        }

        println!(
            "{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_millis()
                - t
        );
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_millis();
        a.add(&b)?;
        println!(
            "{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_millis()
                - t
        );
        Ok(())
    }
}
// cargo test -r --package d2l --lib -- ch3::ch3_1::test::get_start --exact --show-output
