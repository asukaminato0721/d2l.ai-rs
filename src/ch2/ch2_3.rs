#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Tensor as torch};
    #[test]
    fn get_start() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let x = torch::new(&[3.], device)?;
        let y = torch::new(&[2.], device)?;
        println!(
            "{},\n{},\n{},\n{}",
            (&x + &y)?,
            (&x * &y)?,
            (&x / &y)?,
            (&x.pow(&y)?)
        );
        let x = torch::arange(0i64, 3i64, device)?;
        println!("{}", x.i(2)?);

        // emit some contents
        let A = torch::arange(0., 6., device)?.reshape((2, 3))?;
        let B = A.clone();
        println!("{}\n{}", A, (&A + &B)?);
        println!("{}", (&A * &B)?);
        // idk how to mul Tensor with num, but I know how to mul Tensor with Tensor.
        let a = torch::new(&[2i64], device)?;
        let X = torch::arange(0i64, 24i64, device)?.reshape((2, 3, 4))?;
        println!(
            "{} {:?}",
            X.broadcast_add(&a)?,
            X.broadcast_mul(&a)?.shape()
        );
        println!("{:?}\n{}", A.shape(), A.sum_all()?);
        println!("{:?}\n{:?}", A.shape(), A.sum(0)?.shape());
        println!("{:?}\n{:?}", A.shape(), A.sum(1)?.shape());
        println!("{:?}", A.mean(0));
        // 2.3.7
        let sum_A = A.sum_keepdim(1)?;
        println!("{}\n{:?}", &sum_A, sum_A.shape());
        println!("{}", (A.broadcast_div(&sum_A))?);

        println!("{}", A.cumsum(0)?);
        // 2.3.8
        let x = torch::arange(0i64, 3i64, device)?;
        let y = torch::ones(3, DType::I64, device)?;
        println!("{} {}", x, y);
        // TODO: how to torch.dot
        println!("{}", (&x * &y)?.sum_all()?);
        // TODO matrix mul with vec
        // println!("{:?} {:?} {:?}", A.shape(), x.shape(), A.broadcast_matmul(&x));
        let B = torch::ones((3, 4), DType::F64, device)?;
        println!("{}", A.matmul(&B)?);

        let u = torch::new(&[3.0, -4.0], device)?;
        // TODO norm
        println!("{}", u.powf(2.)?.sum_all()?.sqrt()?);

        println!("{}", u.abs()?.sum_all()?);
        println!(
            "{}",
            torch::ones((4, 9), DType::F32, device)?
                .powf(2.)?
                .sum_all()?
                .sqrt()?
        );

        Ok(())
    }
}
