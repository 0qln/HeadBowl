using HeadBowl.TrainingData;
using HeadBowl.Layers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using HeadBowl.Nets;
using HeadBowl.Optimizers;
using Iced.Intel;
using System.Runtime.InteropServices;
using HeadBowl.Helpers;
using HeadBowl.TrainingData.Enviroments;
using HeadBowl.ReinforcementLearning;
using System;
using System.Diagnostics;


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
        const double BETA = 0.9;

        var nn = Net.Build(
            new FullyConnectedLayer<double, Sigmoid>(10, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(20, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(20, Optimizers.Momentum(BETA)),
            new FullyConnectedLayer<double, Sigmoid>(1, Optimizers.Momentum(BETA))
        );

        nn.Load(@"D:\Programmmieren\Projects\HeadBowl\ConsoleApp\bin\Debug\Backups\net-64bit-layers_10_20_20_1-backup-(0).dat");

        double[] GetInputs(Position state, int action)
        {
            var inputs = new double[state.SquareColors.Length + 1];
            Array.Copy(state.SquareColors, inputs, state.SquareColors.Length);
            inputs[state.SquareColors.Length] = action;
            return inputs;
        }

        MonteCarloQLearning<double, Position, int>.QFunction Q = (state, action) =>
        {
            var inputs = GetInputs(state, action);
            var outputs = nn.Forward(inputs);
            return outputs[0];
        };

        for (int i = 0; i < 30; i++)
        {
            Position position = new();
            int us = position.Turn, result;
            int n = 0;
            int[] moves = new int[9];
            
            while (position.GameState == GameStates.Ongoing)
            {
                moves[n] = position.LegalMoves().MaxBy(move => Q(position, move));
                position.MakeMove(moves[n]);
                n++;
            }

            result = position.GameState;

            double reward = result == GameStates.Win[us] ? 1 : result == GameStates.Draw ? 0 : -1;

            for (int k = n - 1; k >= 0; k--)
            {
                int action = moves[k];
                position.UndoMove(action);
                
                if (position.Turn == us)
                {
                    double expected = Q(position, action);
                    nn.Train(GetInputs(position, action), [(1.0d / n) * reward - expected]);
                }
            }

            Debug.Assert(position.Turn == us);

            Console.WriteLine($"{reward - Q(position, moves[0])}");
        }

        nn.Safe("../Backups");
    }

    public static void Quickplay(bool agentBegin, INet<double> nn)
    {
        double[] GetInputs(Position state, int action)
        {
            var inputs = new double[state.SquareColors.Length + 1];
            Array.Copy(state.SquareColors, inputs, state.SquareColors.Length);
            inputs[state.SquareColors.Length] = action;
            return inputs;
        }

        MonteCarloQLearning<double, Position, int>.QFunction Q = (state, action) =>
        {
            var inputs = GetInputs(state, action);
            var outputs = nn.Forward(inputs);
            return outputs[0];
        };

        Position p = new();
        Console.WriteLine(p);

        while (p.GameState == GameStates.Ongoing)
        {
            int move = agentBegin
                ? p.LegalMoves().MaxBy(move => Q(p, move))
                : Utils.StrToSq(Console.ReadLine()!);
            p.MakeMove(move);
            Console.WriteLine(p);

            agentBegin = !agentBegin;
        }
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

