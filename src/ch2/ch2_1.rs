#[cfg(test)]
mod test {
    use candle_core::{DType, Device, IndexOp, Tensor as torch, D};
    #[test]
    fn get_start() -> Result<(), Box<dyn std::error::Error>> {
        let device = &Device::cuda_if_available(0)?;
        let x = torch::arange(0f32, 12., device)?;
        assert_eq!(
            x.to_vec1::<f32>()?,
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]
        );
        assert_eq!(x.elem_count(), 12);
        assert_eq!(x.dims(), [12]);
        let X = x.reshape((3, 4))?;
        assert_eq!(
            X.to_vec2::<f32>()?,
            [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]]
        );
        assert_eq!(
            torch::zeros((2, 3, 4), DType::F32, device)?.to_vec3::<f32>()?,
            [
                [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
                [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]
            ]
        );
        assert_eq!(
            torch::ones((2, 3, 4), DType::F32, device)?.to_vec3::<f32>()?,
            [
                [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]
            ]
        );
        println!("{}", torch::randn(0., 1., (3, 4), device)?);
        assert_eq!(
            torch::new(
                &[[2f32, 1., 4., 3.], [1., 2., 3., 4.], [4., 3., 2., 1.]],
                device
            )?
            .to_vec2::<f32>()?,
            [[2., 1., 4., 3.], [1., 2., 3., 4.], [4., 3., 2., 1.]]
        );
        // to do X[-1]
        assert_eq!(X.i(2)?.to_vec1::<f32>()?, [8., 9., 10., 11.]);
        assert_eq!(
            X.i(1..3)?.to_vec2::<f32>()?,
            [[4., 5., 6., 7.], [8., 9., 10., 11.]]
        );
        // https://github.com/huggingface/candle/issues/1163
        // Tensor can not mut, so ignore
        /// X[:2, :] = 12
        assert_eq!(
            torch::cat(
                &[
                    X.i(..2)?.affine(0., 12.)?.to_dtype(DType::F64)?,
                    X.i(2)?.unsqueeze(0)?.to_dtype(DType::F64)?
                ],
                0
            )?
            .to_vec2::<f64>()?,
            [
                [12., 12., 12., 12.],
                [12., 12., 12., 12.],
                [8., 9., 10., 11.]
            ]
        );
        assert_eq!(
            x.exp()?.to_vec1::<f32>()?,
            [
                1.0, 2.7182817, 7.389056, 20.085537, 54.59815, 148.41316, 403.4288, 1096.6332,
                2980.958, 8103.084, 22026.465, 59874.14
            ]
        );
        let x = torch::new(&[1f32, 2., 4., 8.], device)?;
        let y = torch::new(&[2f32, 2., 2., 2.], device)?;
        // i64 not yet implemented
        assert_eq!(x.add(&y)?.to_vec1::<f32>()?, [3., 4., 6., 10.]);
        assert_eq!(x.sub(&y)?.to_vec1::<f32>()?, [-1., 0., 2., 6.]);
        assert_eq!(x.mul(&y)?.to_vec1::<f32>()?, [2., 4., 8., 16.]);
        assert_eq!(x.div(&y)?.to_vec1::<f32>()?, [0.5, 1.0, 2.0, 4.0]);
        assert_eq!(x.pow(&y)?.to_vec1::<f32>()?, [1., 4., 16., 64.]);
        let X = torch::arange(0., 12., device)?.reshape((3, 4))?;
        let Y = torch::new(
            &[[2.0, 1., 4., 3.], [1., 2., 3., 4.], [4., 3., 2., 1.]],
            device,
        )?;
        assert_eq!(
            torch::cat(&[&X, &Y], 0)?.to_vec2::<f64>()?,
            [
                [0., 1., 2., 3.],
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [2., 1., 4., 3.],
                [1., 2., 3., 4.],
                [4., 3., 2., 1.]
            ]
        );
        assert_eq!(
            torch::cat(&[&X, &Y], 1)?.to_vec2::<f64>()?,
            [
                [0., 1., 2., 3., 2., 1., 4., 3.],
                [4., 5., 6., 7., 1., 2., 3., 4.],
                [8., 9., 10., 11., 4., 3., 2., 1.]
            ]
        );
        assert_eq!(
            X.eq(&Y)?.to_vec2::<u8>()?,
            [[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
        );
        assert_eq!(X.sum_all()?.to_vec0::<f64>()?, 66.);

        // 2.1.4
        let a = torch::arange(0i64, 3i64, device)?.reshape((3, 1))?;
        let b = torch::arange(0i64, 2i64, device)?.reshape((1, 2))?;
        assert_eq!(a.to_vec2::<i64>()?, [[0], [1], [2]]);
        assert_eq!(b.to_vec2::<i64>()?, [[0, 1]]);
        // https://github.com/huggingface/candle/issues/478
        assert_eq!(
            a.broadcast_add(&b)?.to_vec2::<i64>()?,
            [[0, 1], [1, 2], [2, 3]]
        );
        Ok(())
    }
}
