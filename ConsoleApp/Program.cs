using HeadBowl;
using HeadBowl.Training_Data;
using HeadBowl.Layers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

public static class Program
{
    // TODO: Test out using Posit number format [Lombiq Arithmetic]

    public static void Main(string[] args)
    {
        ITrainingData<double> traningData = new Xor<double>();

        INet<double> nn = Net<double>.Build(
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 2),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 300),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 300),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 3),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 1));

        BenchmarkRunner.Run<BenchmarkNet>();

        nn.EnableParallelProcessing = true;

        for (int i = 0;  i < 300000; i++)
        {
            foreach (var data in traningData.Data)
                nn.Train(data.Inputs, data.Expected);


            if (i % 1000 == 0)
                Console.WriteLine(nn.Cost);
        }
    }


}


public class BenchmarkNet
{
    [Benchmark]
    public void BenchmarkLinear()
    {
        INet<double> nn = Net<double>.Build(
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 2),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 300),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 300),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 3),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 1));

        nn.EnableParallelProcessing = false;


        ITrainingData<double> traningData = new Xor<double>();
        for (int i = 0; i < 500; i++)
        {
            foreach (var data in traningData.Data)
                nn.Train(data.Inputs, data.Expected);
        }
    }

    [Benchmark]
    public void BenchmarkParallel()
    {
        INet<double> nn = Net<double>.Build(
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 2),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 300),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 300),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 30),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 3),
            new FullyConnectedLayer<double>(ActivationType.Sigmoid, size: 1));

        nn.EnableParallelProcessing = true;


        ITrainingData<double> traningData = new Xor<double>();
        for (int i = 0; i < 500; i++)
        {
            foreach (var data in traningData.Data)
                nn.Train(data.Inputs, data.Expected);
        }
    }
}