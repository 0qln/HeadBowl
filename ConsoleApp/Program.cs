using HeadBowl.Training_Data;
using HeadBowl.Layers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using HeadBowl.Nets;
using HeadBowl.Optimizers;


//
// TODO:
// Test out using Posit number format [Lombiq Arithmetic]
//


public static class Program
{
    static ITrainingData<double> traningData;
    static INet<double> nn;

    static Program()
    {
        traningData = new Xor<double>();

        nn = Net<double>.Build(
            new FullyConnectedLayer<double, Adam>(ActivationType.Sigmoid, size: 2),
            new FullyConnectedLayer<double, Adam>(ActivationType.Sigmoid, size: 300),
            new FullyConnectedLayer<double, Adam>(ActivationType.Sigmoid, size: 1000),
            new FullyConnectedLayer<double, Adam>(ActivationType.Sigmoid, size: 1));
    }


    public static void Main(string[] args)
    {
    }


    public class BenchmarkNets
    {
        [Benchmark]
        public void NormalLinear()
        {
            nn.ExperimentalFeature = false;
            nn.EnableParallelProcessing = false;

            foreach (var data in traningData.Data)
                nn.Train(data.Inputs, data.Expected);
        }

        //[Benchmark]
        //public void ExperimentalLinear()
        //{
        //    nn.ExperimentalFeature = true;
        //    nn.EnableParallelProcessing = false;

        //    foreach (var data in traningData.Data)
        //        nn.Train(data.Inputs, data.Expected);
        //}

        //[Benchmark]
        //public void NormalParallel()
        //{
        //    nn.ExperimentalFeature = false;
        //    nn.EnableParallelProcessing = true;

        //    foreach (var data in traningData.Data)
        //        nn.Train(data.Inputs, data.Expected);
        //}

        //[Benchmark]
        //public void ExperimentalParallel()
        //{
        //    nn.ExperimentalFeature = true;
        //    nn.EnableParallelProcessing = true;

        //    foreach (var data in traningData.Data)
        //        nn.Train(data.Inputs, data.Expected);
        //}

    }

}

