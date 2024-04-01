//#define LOAD_MNIST
#define RL_MONTECARLO

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


//
// TODO:
// Test out using Posit number format [Lombiq Arithmetic]
//


public static class Program
{

#if LOAD_MNIST
    static Mnist<double> mnist = new(
        testImagesPath: @"D:\Programmmieren\Mnist\t10k-images.idx3-ubyte",
        testLabelsPath: @"D:\Programmmieren\Mnist\t10k-labels.idx1-ubyte",
        trainImagesPath: @"D:\Programmmieren\Mnist\train-images.idx3-ubyte",
        trainLabelsPath: @"D:\Programmmieren\Mnist\train-labels.idx1-ubyte");
#endif

    public static void Main(string[] args)
    {

#if RL_MONTECARLO
        const double BETA = 0.9;

        var nn = Net.Build(
            new FullyConnectedLayer<double, Sigmoid>(10, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(20, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(20, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(1, Optimizers.Momentum(BETA))
        );

        nn.Load(@"..\Backups\net-64bit-layers_10_20_20_1-backup-(0).dat");

        UserQuickplay(false, new Engine<double>(nn), new Position(), Utils.StrToSq);

        //for (int i = 0; i < 30; i++)
        //{
        //    Position position = new();
        //    int us = position.Turn, result;
        //    int n = 0;
        //    int[] moves = new int[9];

        //    while (position.GameState == GameStates.Ongoing)
        //    {
        //        moves[n] = position.LegalMoves().MaxBy(move => Q(position, move));
        //        position.MakeMove(moves[n]);
        //        n++;
        //    }

        //    result = position.GameState;

        //    double reward = result == GameStates.Win[us] ? 1 : result == GameStates.Draw ? 0 : -1;

        //    for (int k = n - 1; k >= 0; k--)
        //    {
        //        int action = moves[k];
        //        position.UndoMove(action);

        //        if (position.Turn == us)
        //        {
        //            double expected = Q(position, action);
        //            nn.Train(GetInputs(position, action), [(1.0d / n) * reward - expected]);
        //        }
        //    }

        //    Debug.Assert(position.Turn == us);

        //    Console.WriteLine($"{reward - Q(position, moves[0])}");
        //}

        //nn.Safe("../Backups");

#endif
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

            foreach (var data in trainingData.Data)
                nn.Train(data.Inputs, data.Expected);
        }

        [Benchmark]
        public void ExperimentalLinear()
        {
            nn.ExperimentalFeature = true;
            nn.EnableParallelProcessing = false;

            foreach (var data in trainingData.Data)
                nn.Train(data.Inputs, data.Expected);
        }

        [Benchmark]
        public void NormalParallel()
        {
            nn.ExperimentalFeature = false;
            nn.EnableParallelProcessing = true;

            foreach (var data in trainingData.Data)
                nn.Train(data.Inputs, data.Expected);
        }

        [Benchmark]
        public void ExperimentalParallel()
        {
            nn.ExperimentalFeature = true;
            nn.EnableParallelProcessing = true;

            foreach (var data in trainingData.Data)
                nn.Train(data.Inputs, data.Expected);
        }

    }

}

