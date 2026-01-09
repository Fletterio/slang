implementing neural;

__include iactivation;

/**
Linear layer parameters packed as a contiguous block:

- weights: Out * In scalars, row-major by output row
  W[row * In + col]
- bias (optional): Out scalars

NOTE: This first implementation targets `InlineVector<T, N>` directly.
We can generalize to other `IVector<T>` implementations once generic dispatch
issues are resolved.
*/
public struct LinearLayer<
    T,
    int In,
    int Out,
    let HasBias : bool = true
>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    public typealias InputVector = InlineVector<T, In>;
    public typealias OutputVector = InlineVector<T, Out>;

    // NOTE: Address type is specialized to `uint` (StructuredBufferStorage<T>.Address).
    // This avoids a current compiler limitation when resolving generic calls while
    // compiling the standard module into `neural.slang-module`.
    public uint weightAddress;
    public uint biasAddress;

    public __init(no_diff uint baseAddress)
    {
        weightAddress = baseAddress;
        if(HasBias)
            biasAddress = baseAddress + uint(Out * In);
    }

    /// Number of scalar parameters consumed by this layer.
    public static int parameterCount()
    {
        return Out * In + (HasBias ? Out : 0);
    }

    /// Returns the base address for the next layer's parameters.
    public static uint nextAddress(no_diff uint baseAddress)
    {
        return baseAddress + uint(parameterCount());
    }
}

/**
Evaluate a LinearLayer.

NOTE: Implemented as a free function (not a struct method) to avoid a current
Slang compiler limitation when resolving generics while compiling standard modules.
*/
[Differentiable]
public InlineVector<T, Out> evalLinearLayer<T, int In, int Out, let HasBias : bool>(
    LinearLayer<T, In, Out, HasBias> layer,
    StructuredBufferStorage<T> storage,
    InlineVector<T, In> input)
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    if(HasBias)
    {
        return input.linearTransform<StructuredBufferStorage<T>, InlineVector<T, Out>>(
            storage, storage, layer.weightAddress, layer.biasAddress);
    }
    else
    {
        return input.linearTransform<StructuredBufferStorage<T>, InlineVector<T, Out>>(
            storage, layer.weightAddress);
    }
}

/**
Feed-forward layer: LinearLayer followed by an activation function.
*/
public struct FFLayer<
    T,
    int In,
    int Out,
    Activation,
    let HasBias : bool = true
>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
    where Activation : IActivation<T>
{
    public typealias InputVector = InlineVector<T, In>;
    public typealias OutputVector = InlineVector<T, Out>;

    public LinearLayer<T, In, Out, HasBias> linear;
    public no_diff Activation activation;

    public __init(no_diff uint baseAddress, no_diff Activation activation)
    {
        linear = LinearLayer<T, In, Out, HasBias>(baseAddress);
        this.activation = activation;
    }

    public static int parameterCount()
    {
        return LinearLayer<T, In, Out, HasBias>.parameterCount();
    }

    public static uint nextAddress(no_diff uint baseAddress)
    {
        return LinearLayer<T, In, Out, HasBias>.nextAddress(baseAddress);
    }

}

/**
Evaluate an FFLayer (linear + activation).
*/
[Differentiable]
public InlineVector<T, Out> evalFFLayer<T, int In, int Out, Activation, let HasBias : bool>(
    FFLayer<T, In, Out, Activation, HasBias> layer,
    StructuredBufferStorage<T> storage,
    InlineVector<T, In> input)
    where T : __BuiltinFloatingPointType
    where T.Differential == T
    where Activation : IActivation<T>
{
    let y = evalLinearLayer<T, In, Out, HasBias>(layer.linear, storage, input);
    return layer.activation.eval<InlineVector<T, Out>>(y);
}

