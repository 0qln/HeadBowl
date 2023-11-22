using HeadBowl.Training_Data;
using HeadBowl.Layers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using HeadBowl.Nets;
using HeadBowl.Optimizers;
using Iced.Intel;


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

        nn = Net.Build(
            new FullyConnectedLayer<double, Sigmoid>(2),
            new FullyConnectedLayer<double, Sigmoid>(3),
            new FullyConnectedLayer<double, Sigmoid>(2),
            new FullyConnectedLayer<double, Sigmoid>(1));

        nn.EnableParallelProcessing = false;
    }


    public static void Main(string[] args)
    {
        var normalNN = Net.Clone(nn);
        var adamNN = Net.Clone(nn);
        Net.SetOptimizer(adamNN, Optimizers.Adam(0.001, 0.9, 0.999, 10E-8));

        CompareNets(normalNN, adamNN);
    }

    public static void CompareNets<TPrecision>(INet<TPrecision> net1, INet<TPrecision> net2)
    {
        var traningData = new Xor<TPrecision>();

        for (int i = 0; i < 50; i++)
        {
            foreach (var data in traningData.Data)
            {
                net1.Train(data.Inputs, data.Expected);
                net2.Train(data.Inputs, data.Expected);
            }

            foreach (var item in net1.Forward(traningData.Data[0].Inputs))
            {
                Console.WriteLine(item);
            }
            foreach (var item in net2.Forward(traningData.Data[0].Inputs))
            {
                Console.WriteLine(item);
            }
            Console.WriteLine();
        }
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

        [Benchmark]
        public void NormalParallel()
        {
            nn.ExperimentalFeature = false;
            nn.EnableParallelProcessing = true;

            foreach (var data in traningData.Data)
                nn.Train(data.Inputs, data.Expected);
        }

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

