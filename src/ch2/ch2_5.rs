#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Result, Tensor as torch, Tensor, Var};
    #[test]
    fn get_start() -> Result<()> {
        let device = &Device::cuda_if_available(0)?;
        /// 2.5.1
        let x = Var::new(&[0., 1., 2., 3.], device)?;
        let x = x.as_tensor();
        assert_eq!(x.to_vec1::<f64>()?, [0., 1., 2., 3.]);
        let y = x.sqr()?.sum_all()?.affine(2., 0.)?;
        assert_eq!(y.to_vec0::<f64>()?, 28.);
        assert_eq!(
            y.backward()?.get(x).unwrap().to_vec1::<f64>()?,
            [0., 4., 8., 12.]
        );
        assert_eq!(
            y.backward()?
                .get(x)
                .unwrap()
                .eq(&(x * 4.)?)?
                .to_vec1::<u8>()?,
            [1, 1, 1, 1]
        );

        let y = x.sum_all()?;
        assert_eq!(
            y.backward()?.get(x).unwrap().to_vec1::<f64>()?,
            [1., 1., 1., 1.]
        );
        /// 2.5.2
        let y = x.sqr()?;
        assert_eq!(
            y.backward()?.get(x).unwrap().to_vec1::<f64>()?,
            [0., 2., 4., 6.]
        );
        /// 2.5.3
        let y = x.mul(&x)?;
        let u = y.detach();
        let z = u.mul(&x)?;
        assert_eq!(
            z.sum_all()?
                .backward()?
                .get(x)
                .unwrap()
                .eq(&u)?
                .to_vec1::<u8>()?,
            [1, 1, 1, 1]
        );
        assert_eq!(
            y.sum_all()?
                .backward()?
                .get(x)
                .unwrap()
                .eq(&x.affine(2., 0.)?)?
                .to_vec1::<u8>()?,
            [1, 1, 1, 1]
        );
        /// 2.5.4
        fn f(a: &Tensor) -> Result<Tensor> {
            let mut b = (a * 2.)?;
            while b.sqr()?.sum_all()?.sqrt()?.to_vec0::<f64>()? < 1000. {
                b = (b * 2.)?;
            }
            return if b.sum_all()?.to_vec0::<f64>()? > 0. {
                Ok(b)
            } else {
                b * 100.
            };
        }
        let a = Var::new(&[1f64], device)?;
        let a = a.as_tensor();
        assert_eq!(f(a)?.backward()?.get(a).unwrap().to_vec1::<f64>()?, [1024.]);
        assert_eq!(
            f(a)?
                .backward()?
                .get(a)
                .unwrap()
                .eq(&f(a)?.div(&a)?)?
                .to_vec1::<u8>()?,
            [1]
        );
        Ok(())
    }
}
