implementing neural;

__include iloss;

/**
Mean Squared Error: mean((predicted - expected)^2)
*/
public struct MeanSquaredError<T> : ILoss<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public T eval<Vector>(Vector predicted, Vector expected)
        where Vector : IVector<T>
    {
        T sum = T(0);
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
        {
            let d = predicted[i] - expected[i];
            sum += d * d;
        }
        return sum / T(Vector.Size);
    }
}

/**
Mean Absolute Error: mean(abs(predicted - expected))
*/
public struct MeanAbsoluteError<T> : ILoss<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    [NoDiffThis, Differentiable]
    public T eval<Vector>(Vector predicted, Vector expected)
        where Vector : IVector<T>
    {
        T sum = T(0);
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
        {
            sum += abs(predicted[i] - expected[i]);
        }
        return sum / T(Vector.Size);
    }
}

/**
Mean Absolute Percentage Error:
mean(abs((predicted - expected) / max(abs(expected), epsilon)))
*/
public struct MeanAbsolutePercentageError<T> : ILoss<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    public T epsilon;
    public __init(T epsilon = T(1e-6)) { this.epsilon = epsilon; }

    [NoDiffThis, Differentiable]
    public T eval<Vector>(Vector predicted, Vector expected)
        where Vector : IVector<T>
    {
        T sum = T(0);
        [ForceUnroll]
        for (int i = 0; i < Vector.Size; i++)
        {
            let denom = max(abs(expected[i]), epsilon);
            sum += abs((predicted[i] - expected[i]) / denom);
        }
        return sum / T(Vector.Size);
    }
}

