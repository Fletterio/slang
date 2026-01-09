implementing neural;

__include iactivation;

/**
Identity activation: returns input unchanged.
*/
public struct IdentityActivation<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public Vector eval<Vector>(Vector input)
        where Vector : IVector<T>
    {
        return input;
    }
}

/**
ReLU activation: max(x, 0).
*/
public struct ReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public Vector eval<Vector>(Vector input)
        where Vector : IVector<T>
    {
        Vector output = Vector();
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
            output[i] = max(input[i], T(0));
        return output;
    }
}

/**
LeakyReLU activation: x < 0 ? alpha*x : x
*/
public struct LeakyReLU<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    public T alpha;

    public __init(T alpha = T(0.01)) { this.alpha = alpha; }

    [NoDiffThis, Differentiable]
    public Vector eval<Vector>(Vector input)
        where Vector : IVector<T>
    {
        Vector output = Vector();
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
        {
            let x = input[i];
            output[i] = (x < T(0)) ? alpha * x : x;
        }
        return output;
    }
}

/**
Sigmoid activation: 1 / (1 + exp(-x))
*/
public struct Sigmoid<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public Vector eval<Vector>(Vector input)
        where Vector : IVector<T>
    {
        Vector output = Vector();
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
        {
            let x = input[i];
            output[i] = T(1) / (T(1) + exp(-x));
        }
        return output;
    }
}

/**
Tanh activation.
*/
public struct TanhActivation<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public Vector eval<Vector>(Vector input)
        where Vector : IVector<T>
    {
        Vector output = Vector();
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
            output[i] = tanh(input[i]);
        return output;
    }
}

/**
Exp activation: exp(x)
*/
public struct ExpActivation<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public Vector eval<Vector>(Vector input)
        where Vector : IVector<T>
    {
        Vector output = Vector();
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
            output[i] = exp(input[i]);
        return output;
    }
}

/**
Sine activation: sin(x)
*/
public struct SineActivation<T> : IActivation<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public Vector eval<Vector>(Vector input)
        where Vector : IVector<T>
    {
        Vector output = Vector();
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
            output[i] = sin(input[i]);
        return output;
    }
}

