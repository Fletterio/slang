implementing neural;

/**
Optimizer interface for updating parameters using gradients.

This interface is intentionally minimal: it defines the scalar update rule.
Higher-level helpers for stepping buffers/storages are provided by concrete optimizers.

@param T Scalar element type (float/half/double).
@param State Optimizer state type (can be a struct).
@category neural
*/
public interface IOptimizer<T, State>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    /**
    Update a single scalar parameter.

    @param[inout] state Optimizer state for this parameter.
    @param[inout] param Parameter value to update.
    @param[inout] grad Gradient value; implementations typically clear it to 0 after the update.
    */
    public void step(inout State state, inout T param, inout T grad);
}

