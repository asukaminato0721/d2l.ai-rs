#[cfg(test)]
mod test {
    use candle_core::{scalar::TensorOrScalar, DType, Device, IndexOp, Result, Tensor as torch};
    #[test]
    fn get_start() -> Result<()> {
        let device = &Device::cuda_if_available(0)?;
        let x = torch::new(&[3.], device)?;
        let y = torch::new(&[2.], device)?;
        assert_eq!(x.add(&y)?.to_vec1::<f64>()?, [5.],);
        assert_eq!(x.mul(&y)?.to_vec1::<f64>()?, [6.]);
        assert_eq!(x.div(&y)?.to_vec1::<f64>()?, [1.5]);
        assert_eq!(x.pow(&y)?.round()?.to_vec1::<f64>()?, [9.]);
        let x = torch::arange(0i64, 3i64, device)?;
        assert_eq!(x.i(2)?.to_scalar::<i64>()?, 2);
        assert_eq!(x.elem_count(), 3);
        assert_eq!(x.dims(), [3]);

        /// 2.3.5
        let A = torch::arange(0., 6., device)?.reshape((2, 3))?;
        let B = A.clone();
        assert_eq!(A.to_vec2::<f64>()?, [[0., 1., 2.], [3., 4., 5.]],);
        assert_eq!(A.add(&B)?.to_vec2::<f64>()?, [[0., 2., 4.], [6., 8., 10.]]);
        assert_eq!(A.mul(&B)?.to_vec2::<f64>()?, [[0., 1., 4.], [9., 16., 25.]]);
        let a = 2.;
        let X = torch::arange(0i64, 24i64, device)?.reshape((2, 3, 4))?;
        assert_eq!(
            X.affine(1., a)?.to_vec3::<i64>()?,
            [
                [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]],
                [[14, 15, 16, 17], [18, 19, 20, 21], [22, 23, 24, 25]]
            ]
        );
        assert_eq!(X.affine(a, 0.)?.dims(), [2, 3, 4]);
        assert_eq!(A.dims(), [2, 3]);
        assert_eq!(A.sum_all()?.get(0)?.to_scalar::<f64>()?, 15.);

        assert_eq!(A.sum(0)?.dims(), [3]);
        assert_eq!(A.sum(1)?.dims(), [2]);
        assert_eq!(A.mean(0)?.to_vec1::<f64>()?, [1.5, 2.5, 3.5]);
        // 2.3.7
        let sum_A = A.sum_keepdim(1)?;
        assert_eq!(sum_A.to_vec2::<f64>()?, [[3.], [12.]]);
        assert_eq!(sum_A.dims(), [2, 1]);
        assert_eq!(
            A.broadcast_div(&sum_A)?.to_vec2::<f64>()?,
            [
                [0.0, 0.3333333333333333, 0.6666666666666666],
                [0.25, 0.3333333333333333, 0.4166666666666667]
            ]
        );
        assert_eq!(A.cumsum(0)?.to_vec2::<f64>()?, [[0., 1., 2.], [3., 5., 7.]]);
        // 2.3.8
        let x = torch::arange(0., 3., device)?;
        let y = torch::ones(3, DType::F64, device)?;
        assert_eq!(x.to_vec1::<f64>()?, [0., 1., 2.]);
        assert_eq!(y.to_vec1::<f64>()?, [1., 1., 1.]);
        assert_eq!(x.mul(&y)?.sum_all()?.get(0)?.to_scalar::<f64>()?, 3.);
        assert_eq!(A.dims(), [2, 3]);
        assert_eq!(x.dims(), [3]);
        // this is torch.mv , so tedious :(
        assert_eq!(
            A.matmul(&x.unsqueeze(0)?.t()?)?
                .flatten_all()?
                .to_vec1::<f64>()?,
            [5., 14.]
        );
        let B = torch::ones((3, 4), DType::F64, device)?;
        assert_eq!(
            A.matmul(&B)?.to_vec2::<f64>()?,
            [[3., 3., 3., 3.], [12., 12., 12., 12.]]
        );

        let u = torch::new(&[3.0, -4.0], device)?;
        // TODO norm
        assert_eq!(
            u.powf(2.)?.sum_all()?.sqrt()?.get(0)?.to_scalar::<f64>()?,
            5.
        );
        assert_eq!(u.abs()?.sum_all()?.get(0)?.to_scalar::<f64>()?, 7.);
        assert_eq!(
            torch::ones((4, 9), DType::F64, device)?
                .powf(2.)?
                .sum_all()?
                .sqrt()?
                .get(0)?
                .to_scalar::<f64>()?,
            6.
        );
        Ok(())
    }
}
