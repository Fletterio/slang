implementing neural;

/**
Activation function interface for neural network operations.
Defines a differentiable mapping from an input vector to an output vector of the same shape.

@param T Scalar element type (float/half/double).
@category neural
*/
public interface IActivation<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    /// Apply activation function element-wise.
    [NoDiffThis, Differentiable]
    public Vector eval<Vector>(Vector input)
        where Vector : IVector<T>;
}

