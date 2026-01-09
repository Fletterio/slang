implementing neural;

__include ioptimizer;

/**
Stateless SGD optimizer.
*/
public struct SGD<T> : IOptimizer<T, int>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    public T learningRate;

    public __init(T learningRate = T(1e-3)) { this.learningRate = learningRate; }

    public void step(inout int state, inout T param, inout T grad)
    {
        state = state; // unused
        param -= learningRate * grad;
        grad = T(0);
    }

    /**
    Step a contiguous range of parameters stored in IStorage.
    @param count number of scalar elements.
    */
    public void stepStorage<Storage>(
        Storage params,
        Storage grads,
        no_diff Storage.Address paramsBase,
        no_diff Storage.Address gradsBase,
        int count)
        where Storage : IStorage<T>
        where Storage.Differential : IStorage<T.Differential>
        where Storage.Address == Storage.Differential.Address
    {
        [MaxIters(1024)]
        for (int i = 0; i < count; i++)
        {
            let pAddr = Storage.getOffset(paramsBase, i);
            let gAddr = Storage.getOffset(gradsBase, i);
            var p = params.read(pAddr);
            var g = grads.read(gAddr);
            p -= learningRate * g;
            params.write(pAddr, p);
            grads.write(gAddr, T(0));
        }
    }
}

/**
Adam optimizer state for a single parameter.
*/
public struct AdamState<T>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    public T m;
    public T v;
    public int iteration;

    public __init(T initialParam = T(0))
    {
        m = T(0);
        v = T(0);
        iteration = 0;
        initialParam = initialParam; // unused, but handy for symmetry with other frameworks
    }
}

/**
Adam optimizer (per-parameter state stored as a struct).

Note: storing `AdamState<T>` in a buffer is up to the user (e.g. RWStructuredBuffer<AdamState<T>>),
because `IStorage<T>` currently only supports floating-point element types (not structs).
*/
public struct Adam<T> : IOptimizer<T, AdamState<T>>
    where T : __BuiltinFloatingPointType
    where T.Differential == T
{
    public T beta1;
    public T beta2;
    public T epsilon;
    public T learningRate;

    public __init(
        T learningRate = T(1e-3),
        T beta1 = T(0.9),
        T beta2 = T(0.999),
        T epsilon = T(1e-8))
    {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public void step(inout AdamState<T> state, inout T param, inout T grad)
    {
        state.iteration++;

        state.m = beta1 * state.m + (T(1) - beta1) * grad;
        state.v = beta2 * state.v + (T(1) - beta2) * grad * grad;

        // Bias correction
        let t = T(state.iteration);
        let mHat = state.m / (T(1) - pow(beta1, t));
        let vHat = state.v / (T(1) - pow(beta2, t));

        param -= learningRate * mHat / (sqrt(max(T(0), vHat)) + epsilon);
        grad = T(0);
    }

    /**
    Adam step for parameters in storage using two extra moment buffers (m and v), plus an externally-managed iteration.

    Layout expectation:
    - `mStorage` holds count elements (m per parameter)
    - `vStorage` holds count elements (v per parameter)
    */
    public void stepStorage<Storage>(
        Storage params,
        Storage grads,
        Storage mStorage,
        Storage vStorage,
        no_diff Storage.Address paramsBase,
        no_diff Storage.Address gradsBase,
        no_diff Storage.Address mBase,
        no_diff Storage.Address vBase,
        int count,
        int iteration)
        where Storage : IStorage<T>
        where Storage.Differential : IStorage<T.Differential>
        where Storage.Address == Storage.Differential.Address
    {
        let t = T(iteration);
        let invBias1 = T(1) / (T(1) - pow(beta1, t));
        let invBias2 = T(1) / (T(1) - pow(beta2, t));

        [MaxIters(1024)]
        for (int i = 0; i < count; i++)
        {
            let pAddr = Storage.getOffset(paramsBase, i);
            let gAddr = Storage.getOffset(gradsBase, i);
            let mAddr = Storage.getOffset(mBase, i);
            let vAddr = Storage.getOffset(vBase, i);

            var p = params.read(pAddr);
            var g = grads.read(gAddr);
            var m = mStorage.read(mAddr);
            var v = vStorage.read(vAddr);

            m = beta1 * m + (T(1) - beta1) * g;
            v = beta2 * v + (T(1) - beta2) * g * g;

            let mHat = m * invBias1;
            let vHat = v * invBias2;

            p -= learningRate * mHat / (sqrt(max(T(0), vHat)) + epsilon);

            params.write(pAddr, p);
            grads.write(gAddr, T(0));
            mStorage.write(mAddr, m);
            vStorage.write(vAddr, v);
        }
    }
}

