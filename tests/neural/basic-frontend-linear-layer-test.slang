// Basic test for builtin neural "frontend" APIs (layers + activations + losses + optimizers).
//
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-vk -compute -shaderobj -xslang -experimental-feature -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-dx12 -compute -shaderobj -profile cs_6_6 -xslang -experimental-feature -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-mtl -compute -shaderobj -output-using-type -xslang -experimental-feature
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-cuda -compute -shaderobj -output-using-type -capability cuda_sm_7_0 -xslang -experimental-feature

import neural;

// Parameters: LinearLayer<float, 4, 2, ..., HasBias=true>
// weights = 2x4 matrix, bias = 2
//TEST_INPUT: ubuffer(data=[1 2 3 4  5 6 7 8  9 10], stride=4):name=parametersFloat
RWStructuredBuffer<float> parametersFloat;

//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):name=params
RWStructuredBuffer<float> params;

// grads for params (same size)
//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):name=grads
RWStructuredBuffer<float> grads;

// optimizer moments (m, v), each count=10 for simplicity (only first 10 used)
//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):name=mom1
RWStructuredBuffer<float> mom1;
//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):name=mom2
RWStructuredBuffer<float> mom2;

//TEST_INPUT: ubuffer(data=[0 0 0], stride=4):out,name=testResult
RWStructuredBuffer<uint> testResult;

typealias Storage = StructuredBufferStorage<float>;

[Differentiable]
float lossFn(Storage storage, no_diff Storage.Address baseAddr)
{
    // Input x = [1,2,3,4]
    float[4] xArr = {1.0, 2.0, 3.0, 4.0};
    let x = no_diff InlineVector<float, 4>(xArr);

    // Layer: y = W*x + b, then ReLU
    typealias XVec = InlineVector<float, 4>;
    typealias YVec = InlineVector<float, 2>;
    no_diff let layer = FFLayer<float, 4, 2, ReLU<float>, true>(baseAddr, ReLU<float>());
    let y = evalFFLayer<float, 4, 2, ReLU<float>, true>(layer, storage, x);

    // Expected (target) for loss
    float[2] tArr = { 39.0, 80.0 }; // matches W*x+b with given params
    let t = no_diff YVec(tArr);

    return MeanSquaredError<float>().eval<YVec>(y, t);
}

bool forwardAndLossTest()
{
    // Copy parametersFloat -> params
    for(int i=0;i<10;i++) params[i] = parametersFloat[i];

    let storage = Storage(params);
    let l = lossFn(storage, 0);
    return l == 0.0;
}

bool backwardGradTest()
{
    // grads zeroed already
    let storage = Storage(params);
    let dStorage = Storage(grads);
    var dPair = DifferentialPtrPair<Storage>(storage, dStorage);

    // Because loss is 0 at the exact target, gradients should stay 0
    bwd_diff(lossFn)(dPair, 0, float(1.0));

    bool ok = true;
    for(int i=0;i<10;i++) ok = ok && (grads[i] == 0.0);
    return ok;
}

bool optimizerSmokeTest()
{
    // Put a non-zero grad on first element, then run one SGD step and verify param changed.
    grads[0] = 1.0;

    let pStorage = Storage(params);
    let gStorage = Storage(grads);

    SGD<float> sgd = SGD<float>(0.1);
    int dummyState = 0;
    // scalar path
    float p0 = pStorage.read(0);
    float g0 = gStorage.read(0);
    sgd.step(dummyState, p0, g0);
    pStorage.write(0, p0);
    gStorage.write(0, g0);

    return (params[0] == parametersFloat[0] - 0.1) && (grads[0] == 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    testResult[0] = forwardAndLossTest();
    testResult[1] = backwardGradTest();
    testResult[2] = optimizerSmokeTest();

    // BUFFER: 1
    // BUFFER: 1
    // BUFFER: 1
}

