implementing neural;

/**
Loss function interface for neural network training.
Computes a scalar loss value from predicted and expected vectors.

@param T Scalar element type (float/half/double).
@category neural
*/
public interface ILoss<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public T eval<Vector>(Vector predicted, Vector expected)
        where Vector : IVector<T>;
}

