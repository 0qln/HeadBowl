using HeadBowl.Training_Data;
using HeadBowl.Layers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using HeadBowl.Nets;
using HeadBowl.Optimizers;
using Iced.Intel;
using System.Runtime.InteropServices;
using HeadBowl.Helpers;


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
        //var normalNN = Net.Clone(nn);
        //var optimNN = Net.Clone(nn);
        //Net.SetOptimizer(optimNN, Optimizers.Momentum(0.9));
        //CompareNets(normalNN, optimNN, 500, 14);

        const double BETA = 0.9;
        const int MSG = 30;

        BenchmarkNet(epochs: MSG * 5, messages: MSG, net: Net.Build(
            new FullyConnectedLayer<double, Sigmoid>(2, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(300, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(200, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(300, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(200, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(1, Optimizers.Momentum(BETA))
        ));
    }

    public static void BenchmarkNet<TPrecision>(INet<TPrecision> net, int epochs, int messages)
    {
        var traningData = new Xor<TPrecision>();

        for (int i = 0; i < epochs; i++)
        {
            var batch = Choose.Cycle(traningData);

            net.Train(batch.Inputs, batch.Expected);

            if (i % (epochs / messages) == 0)
                foreach (var item in net.Forward(batch.Inputs)) 
                    Console.WriteLine(item);
        }
    }

    public static void CompareNets<TPrecision>(INet<TPrecision> net1, INet<TPrecision> net2, int epochs, int messages)
    {
        var traningData = new Xor<TPrecision>();

        for (int i = 0; i < epochs; i++)
        {
            var batch = Choose.Random(traningData);

            net1.Train(batch.Inputs, batch.Expected);
            net2.Train(batch.Inputs, batch.Expected);

            if (i % (epochs / messages) == 0)
            {
                foreach (var item in net1.Forward(batch.Inputs)) Console.WriteLine(item);
                foreach (var item in net2.Forward(batch.Inputs)) Console.WriteLine(item);
                
                Console.WriteLine();
            }
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

