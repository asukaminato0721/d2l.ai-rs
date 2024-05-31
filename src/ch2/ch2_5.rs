#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Tensor as torch, Tensor, Var};
    #[test]
    fn get_start() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::Cpu;
        let x = Var::new(&[0., 1., 2., 3.], device)?;
        let x = x.as_tensor();
        println!("{x}");
        let y = x.powf(2.)?.sum_all()?.affine(2., 0.)?;

        println!("{}", y);
        println!("{}", y.backward()?.get(&x).unwrap());
        println!("{}", y.backward()?.get(&x).unwrap().eq(&x.affine(4., 0.)?)?);

        let y = x.sum_all()?;
        println!("{}", y.backward()?.get(&x).unwrap());
        let y = x.powf(2.)?;
        println!("{}", y.backward()?.get(&x).unwrap());
        fn f(a: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
            let mut b = a.affine(2f64, 0f64).unwrap();
            while b.powf(2.)?.sum_all()?.sqrt()?.to_scalar::<f64>()? < 1000. {
                b = b.affine(2., 0.)?;
            }
            return if b.sum_all()?.to_scalar::<f64>()? > 0. {
                Ok(b)
            } else {
                Ok(b.affine(100., 0.)?)
            };
        }
        let a = Var::new(&[1f64], device)?;
        let a = a.as_tensor();
        println!("{}", f(a)?.backward()?.get(&a).unwrap());
        println!(
            "{}",
            f(a)?.backward()?.get(&a).unwrap().eq(&f(a)?.div(&a)?)?
        );
        Ok(())
    }
}
