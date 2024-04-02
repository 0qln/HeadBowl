//#define RL_MONTECARLO

using HeadBowl.TrainingData;
using HeadBowl.Layers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using HeadBowl.Nets;
using HeadBowl.Optimizers;
using Iced.Intel;
using System.Runtime.InteropServices;
using HeadBowl.Helpers;
using HeadBowl.ReinforcementLearning;
using System;
using System.Diagnostics;
using HeadBowl.TrainingData.TicTacToe;
using System.Numerics;


//
// TODO:
// Test out using Posit number format [Lombiq Arithmetic]
//


public static class Program
{
    public static void Main(string[] args)
    {
        var optimizer = Optimizers.None<double>();

        var nn = Net.Build(
            new FullyConnectedLayer<double, Sigmoid>(28 * 28, optimizer),
            new FullyConnectedLayer<double, Sigmoid>(28 * 28 / 2, optimizer),
            new FullyConnectedLayer<double, Sigmoid>(28 * 28 / 4, optimizer),
            new FullyConnectedLayer<double, Sigmoid>(28 * 28 / 8, optimizer),
            new FullyConnectedLayer<double, Sigmoid>(28 * 28 / 16, optimizer),
            new FullyConnectedLayer<double, Sigmoid>(1, optimizer)
        );
        nn.EnableParallelProcessing = true;

        BenchmarkNetMnist(nn, 10_000, 100);


#if RL_MONTECARLO

        nn.Load(@"..\..\Release\Backups\net-64bit-layers_10_200_300_200_200_1-backup-(8).dat");

        UserQuickplay(false, new Engine<double>(nn), new Position(), Utils.StrToSq);

        //for (int i = 0; i < 10000; i++)
        //{
        //    SelfPlay(new Engine<double>(nn), new Position());

        //    if (i % 100 == 0)
        //    {
        //        nn.Safe("../Backups");
        //    }
        //}


#endif
    }

    public static void SelfPlay<TAction, TPrecision, TEnviroment>(
            IAgent<TPrecision, TEnviroment, TAction> agent,
            TEnviroment env)
        where TEnviroment : ITwoAgentEnviroment<TAction>
        where TPrecision : struct, ISubtractionOperators<TPrecision, TPrecision, TPrecision>
    {
        List<TAction> takenActions = [];
        int us = env.Turn;

        // Play the game according to the agents policy.
        while (!env.Terminal)
        {
            takenActions.Add(env.LegalActions().MaxBy(action => agent.Q(env, action))
                ?? throw new Exception("No more moves in the env., even tho the env. is not terminal."));
            env.MakeAction(takenActions.Last());
        }

        int n = takenActions.Count;

        int finalReward = env.Reward;
        foreach (var action in takenActions.Reverse<TAction>())
        {
            env.UndoAction(action);
            MonteCarloQLearning<TPrecision>.Update(agent, (dynamic)finalReward, env, action, n);
            finalReward = -finalReward;
        }
    }

    public static void UserQuickplay<TAction, TPrecision, TEnviroment>(
            bool agentBegin,
            IAgent<TPrecision, TEnviroment, TAction> agent,
            TEnviroment env,
            Func<string, TAction> strToAct)
        where TEnviroment : IEnviroment<TAction>
        where TPrecision : struct
    {
        Console.WriteLine(env);
        while (!env.Terminal)
        {
            env.MakeAction(agentBegin
                ? env.LegalActions().MaxBy(move => agent.Q(env, move))
                    ?? throw new Exception("No more moves in the env., even tho the env. is not terminal.")
                : strToAct(Console.ReadLine()!));

            Console.WriteLine(env);
            agentBegin = !agentBegin;
        }
        Console.WriteLine("Quickplay finished.");
    }

    public static void BenchmarkNetXor<TPrecision>(INet<TPrecision> net, int epochs, int messages)
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

    public static void BenchmarkNetMnist<TPrecision>(INet<TPrecision> net, int epochs, int messages)
        where TPrecision : struct
    {
        Mnist<double> mnist = new(
            testImagesPath: @"D:\Programmmieren\Mnist\t10k-images.idx3-ubyte",
            testLabelsPath: @"D:\Programmmieren\Mnist\t10k-labels.idx1-ubyte",
            trainImagesPath: @"D:\Programmmieren\Mnist\train-images.idx3-ubyte",
            trainLabelsPath: @"D:\Programmmieren\Mnist\train-labels.idx1-ubyte"
        );

        var data = Choose.Random(mnist as ITrainingData<TPrecision>);

        for (int i = 0; i < epochs; i++)
        {
            //var train = Choose.Random(mnist as ITrainingData<TPrecision>);
            net.Train(data.Inputs, data.Expected);

            if (i % (epochs / messages) == 0)
            {
                Console.WriteLine("Testing Results [Output | Expected]: ");
                //for (int j = 0; j < 10; j++)
                {
                    //var test = Choose.Random(mnist as ITestingData<TPrecision>);
                    PrintMnistInput(data.Inputs);
                    Console.WriteLine($"{data.Expected[0]} | {net.Forward(data.Inputs)[0]}");
                }                    
            }
        }
    }

    static readonly string gradient = String.Concat(@"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,""^`'. ".Reverse());
    public static void PrintMnistInput<T>(T[] inputs)
        where T : struct
    {
        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
                Console.Write(new string(gradient[(int)((dynamic)inputs[x + 28 * y] / 255 * gradient.Length) % gradient.Length], 2));
            Console.WriteLine();
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



    public class BenchmarkNets<TPrecision>
    {
        public IEnumerable<ITrainingData<TPrecision>> tdSource => [
                new Xor<TPrecision>()
            ];

        public IEnumerable<INet<TPrecision>> nnSource => [
                Net.Build(
                    new FullyConnectedLayer<TPrecision, Sigmoid>(trainingData.InputSize),
                    new FullyConnectedLayer<TPrecision, Sigmoid>(100),
                    new FullyConnectedLayer<TPrecision, Sigmoid>(100),
                    new FullyConnectedLayer<TPrecision, Sigmoid>(trainingData.OutputSize)),

                Net.Build(
                    new FullyConnectedLayer<TPrecision, Sigmoid>(trainingData.InputSize),
                    new FullyConnectedLayer<TPrecision, Sigmoid>(10),
                    new FullyConnectedLayer<TPrecision, Sigmoid>(10),
                    new FullyConnectedLayer<TPrecision, Sigmoid>(trainingData.OutputSize)),
            ];

        [ParamsSource(nameof(nnSource))]
        public INet<TPrecision> nn = null!;

        [ParamsSource(nameof(tdSource))]
        public ITrainingData<TPrecision> trainingData = null!;

        [Benchmark]
        public void NormalLinear()
        {
            nn.ExperimentalFeature = false;
            nn.EnableParallelProcessing = false;

            foreach (var data in trainingData.TrainingData)
                nn.Train(data.Inputs, data.Expected);
        }

        [Benchmark]
        public void ExperimentalLinear()
        {
            nn.ExperimentalFeature = true;
            nn.EnableParallelProcessing = false;

            foreach (var data in trainingData.TrainingData)
                nn.Train(data.Inputs, data.Expected);
        }

        [Benchmark]
        public void NormalParallel()
        {
            nn.ExperimentalFeature = false;
            nn.EnableParallelProcessing = true;

            foreach (var data in trainingData.TrainingData)
                nn.Train(data.Inputs, data.Expected);
        }

        [Benchmark]
        public void ExperimentalParallel()
        {
            nn.ExperimentalFeature = true;
            nn.EnableParallelProcessing = true;

            foreach (var data in trainingData.TrainingData)
                nn.Train(data.Inputs, data.Expected);
        }

    }

}

