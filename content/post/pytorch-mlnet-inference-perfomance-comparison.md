---
author: "Valerii Boldakov"
title: "PyTorch and ML.NET Inference Performance Comparison"
draft: false
date: 2022-02-06
description: "Let's say you have a working and a developer-friendly .NET ecosystem. 
There are a lot of services and your team doesn't cherish the idea of having a service built without .NET. 
Additionally, there is a pending request to develop software to serve some machine learning models."
tags: ["machine-learning", "pytorch", "ml.net"]
archives: ["2022/02"]
image: "img/pytorch-mlnet-inference-perfomance-comparison/1.png"
---

Let's say you have a working and a developer-friendly .NET ecosystem. 
There are a lot of services and your team doesn't cherish the idea of having a service built without .NET. 
Additionally, there is a pending request to develop software to serve some machine learning models.

Most of the machine learning models are created with Python and libraries like scikit-learn, PyTorch, or TensorFlow. Here come the questions. Should the team just implement a Python service with the abovementioned libraries? Perhaps using ML.NET for the sake of the ecosystem consistency would be a better solution?
I will try to find some rationale behind these solutions. To resolve it I made inferences of the ResNet18 deep learning model using PyTorch and ML.NET and compared their performance.

### PyTorch Performance

PyTorch is a widely known open source library for deep learning. It's no wonder that most of the researchers use it to create a state of the art models. It's a popular choice for Python developers to evaluate acquired models.
To get measurements of the models I used the following environment and hardware:

- GeForce GTX 1660.
- AMD Ryzen 5 3600.
- Ubuntu 18.04.
- CUDA Toolkit 10.1.
- cuDNN 7.0.
- torch 1.7.1.
- torchvision 0.8.2.

The ResNet18 model is obtained from the PyTorch Hub. In order to get the inference times, I made 10000 inferences with GPU and CPU. At this point, my interest didn't lie in the output of the model so using a random tensor as an input sufficed. It's also the reason why I didn't scale the input tensor. I used this code to generate inference time:

```python
resnet_gpu_model = torch.jit.trace(torchvision.models.resnet18(pretrained=True).eval().cuda(), torch.randn(1, 3, 256, 260).cuda())
resnet_cpu_model = torch.jit.trace(torchvision.models.resnet18(pretrained=True).eval().cpu(), torch.randn(1, 3, 256, 260))
gpu_inference_time = []
cpu_inference_time = []
for _ in range(10000):
    input_tensor = torch.randn(3, 256, 260)
    input_batch = input_tensor.unsqueeze(0)
    gpu_input_batch = input_batch.cuda()
    with torch.no_grad():
        start = timer()
        gpu_output = resnet_gpu_model(gpu_input_batch)
        end = timer()
        gpu_inference_time.append(end - start)
        start = timer()
        cpu_output = resnet_cpu_model(input_batch)
        end = timer()
        cpu_inference_time.append(end - start)
```

The following distributions were received after the code execution:

![](/img/pytorch-mlnet-inference-perfomance-comparison/2.png)

The mean inference time for CPU was 0.026 seconds and 0.001 seconds for GPU. Their standard deviations were 0.003 and 0.0001 respectively. GPU execution was roughly 10 times faster, which is what was expected.

### ML.NET Introduction

ML.NET is a machine learning framework built for .NET developers. It has a built-in AutoML allowing you to avoid choosing the best type of model manually. However, it lacks the possibility to define and train your own neural networks. You need to import your model in ONNX format from PyTorch with the following code:

```python
torch.onnx.export(resnet_cpu_model,
                  input_batch,
                  "resnet18.onnx",
                  export_params=True,
                  do_constant_folding=True,
                  inpput_names = ut_names = ['input'],
                  out['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})
```

ML.NET heavily relies on the ONNX Runtime accelerator to make use of the deep learning models and all the inferences are made using ONNX Runtime.

To conduct ML.NET inferences I used the same hardware but another environment because of ML.NET requirements:

- Ubuntu 18.04.
- CUDA Toolkit 10.2.
- cuDNN 8.

This ML.NET code will have a more thorough description because it’s much less popular than PyTorch. At the first step, we need to install NuGET packages with ML.NET and ONNX Runtime:

- Microsoft.ML 1.5.4.
- Microsoft.ML.OnnxRuntime.Gpu 1.6.0.
- Microsoft.ML.OnnxTransformer 1.5.4.

Before trying to make any inference we need to create two classes representing the input and output of the model:

```cs
public class ImageLabels
    {
        [ColumnName("output")] public float[] Labels { get; set; }
    }

    public class PixelValues
    {
        public const int ChannelAmount = 3;
        public const int ImageWidth = 256;
        public const int ImageHeight = 260;

        [VectorType(3, 256, 260)]
        [ColumnName("input")]
        public float[] Values { get; set; }
    }
```

The most important thing is to create an MLContext object. This object is central to the ML.NET framework. Every inference and preprocessing operations are defined with it. It has a nice feature to set a random seed to make operations deterministic.

As the next step, you can define a model estimator, fit it to let it know about the input data scheme, and finally create a prediction engine. This can be achieved with the following code:

```cs
var mlContext = new MLContext();
var gpuModelEstimator = mlContext.Transforms.ApplyOnnxModel(outputColumnName: "output", inputColumnName: "input", modelFile: OnnxModelPath, gpuDeviceId: 0);
var cpuModelEstimator = mlContext.Transforms.ApplyOnnxModel(outputColumnName: "output", inputColumnName: "input", modelFile: OnnxModelPath);
var data = mlContext.Data.LoadFromEnumerable(new List<PixelValues>());
var gpuModel = gpuModelEstimator.Fit(data);
var cpuModel = cpuModelEstimator.Fit(data);
var gpuPredictionEngine = mlContext.Model.CreatePredictionEngine<PixelValues, ImageLabels>(gpuModel);
var cpuPredictionEngine = mlContext.Model.CreatePredictionEngine<PixelValues, ImageLabels>(cpuModel);
```

Finally, you can create some input data, make inferences, and look at your estimation:

```cs
var randNum = new Random();
var gpuInferenceTime = new List<double>();
var cpuInferenceTime = new List<double>();
for (var i = 0; i < 10000; ++i)
{
    var value = new PixelValues
    {
        Values = Enumerable
            .Repeat(0, PixelValues.ChannelAmount * PixelValues.ImageHeight * PixelValues.ImageWidth)
            .Select(_ => (float) randNum.NextDouble()).ToArray()
    };
    var sw = new Stopwatch();
    sw.Start();
    var labels = gpuPredictionEngine.Predict(value).Labels;
    sw.Stop();
    gpuInferenceTime.Add(sw.Elapsed.TotalSeconds);
    sw.Start();
    labels = cpuPredictionEngine.Predict(value).Labels;
    sw.Stop();
    cpuInferenceTime.Add(sw.Elapsed.TotalSeconds);
}
```

This resulted in the following distributions:

![](/img/pytorch-mlnet-inference-perfomance-comparison/3.png)

Mean inference time for CPU was 0.016 seconds and 0.005 seconds for GPU with standard deviations 0.0029 and 0.0007 respectively.

### Conclusion

Upon measuring the performance of each framework we can summarize it with the following graph:

![](/img/pytorch-mlnet-inference-perfomance-comparison/4.png)

ML.NET can evaluate deep learning models with a decent speed and is faster than PyTorch using CPU. It can be a dealbreaker for production use. With ML.NET you can have all the advantages of the .NET ecosystem, fast web servers like Kestrel, and easily-maintainable object-oriented code.

Yet there are some drawbacks to take note of. PyTorch GPU inference is faster. Using ML.NET for some tasks which rely heavily on GPU computation may be slower. Before evaluating a model you would often need to preprocess your data. Even though ML.NET has a rich library of different transformers for data preprocessing, it may be a complex task to reproduce the same preprocessing in your .NET software.

Deciding what framework to use depends on your current situation. How much time do you have to develop the service? How important performance is for you? Does data preprocessing depend on some Python library? You need the answer to all of these questions while keeping in mind ML.NET and PyTorch framework performance.
