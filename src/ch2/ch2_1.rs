#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Tensor as torch};
    #[test]
    fn get_start() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let x = torch::arange(0f32, 12., &Device::Cpu)?;
        println!("{x}");
        println!("{:?}", x.shape());
        let mut X = x.reshape((3, 4))?;
        println!("{X}");
        println!("{}", torch::zeros((2, 3, 4), DType::F32, &Device::Cpu)?);
        println!("{}", torch::ones((2, 3, 4), DType::F32, &Device::Cpu)?,);
        println!("{}", torch::randn(0., 1., (3, 4), &Device::Cpu)?);
        println!(
            "{}",
            torch::new(
                &[[2f32, 1., 4., 3.], [1., 2., 3., 4.], [4., 3., 2., 1.]],
                &Device::Cpu
            )?
        );
        // to do X[-1]
        println!("{}\n{}", X.i(2)?, X.i(1..3)?);
        // https://github.com/huggingface/candle/issues/1163
        // Tensor can not mut, so ignore

        println!("{}", torch::exp(&x)?);
        let x = torch::new(&[1f32, 2., 4., 8.], device)?;
        let y = torch::new(&[2f32, 2., 2., 2.], device)?;
        // i64 not yet implemented
        println!(
            "+{}\n-{}\n*{}\n/{}\n**{}\n",
            (&x + &y)?,
            (&x - &y)?,
            (&x * &y)?,
            (&x / &y)?,
            x.pow(&y)?
        );
        X = torch::arange(0., 12., device)?.reshape((3, 4))?;
        let Y = torch::new(
            &[[2.0, 1., 4., 3.], [1., 2., 3., 4.], [4., 3., 2., 1.]],
            device,
        )?;
        println!(
            "{} {}",
            torch::cat(&[&X, &Y], 0)?,
            torch::cat(&[&X, &Y], 1)?
        );
        println!("{}", X.eq(&Y)?);
        println!("{}", X.sum_all()?);

        // 2.1.4
        let a = torch::arange(0i64, 3i64, device)?.reshape((3, 1))?;
        let b = torch::arange(0i64, 2i64, device)?.reshape((1, 2))?;
        println!("{}\n{}", a, b);
        // https://github.com/huggingface/candle/issues/478
        println!("{}", a.broadcast_add(&b)?);
        Ok(())
    }
}
