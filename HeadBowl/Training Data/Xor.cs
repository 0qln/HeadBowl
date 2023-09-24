using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Training_Data
{
    public interface ITrainingData<T>
    {
        public int InputSize { get; }
        public int OutputSize { get; }
        public int Samples { get; }

        public TrainingDataInstance<T>[] Data { get; }
    }

    public class TrainingDataInstance<T>
    {
        public T[] Inputs, Expected;

        public TrainingDataInstance(T[] inputs, T[] expected)
        {
            Inputs = inputs;
            Expected = expected;
        }
        public TrainingDataInstance(int inputSize, int outputSize)
        {
            Inputs = new T[inputSize];
            Expected = new T[outputSize];
        }
    }

    public class Xor<T> : ITrainingData<T>
    {
        public int InputSize { get; } = 2;
        public int OutputSize { get; } = 1;
        public int Samples => Data.Length;

        public TrainingDataInstance<T>[] Data => new TrainingDataInstance<T>[]
        {
            new TrainingDataInstance<T>(new T[]{ (dynamic)0, (dynamic)0 }, new T[]{ (dynamic)0 } ),
            new TrainingDataInstance<T>(new T[]{ (dynamic)1, (dynamic)0 }, new T[]{ (dynamic)1 } ),
            new TrainingDataInstance<T>(new T[]{ (dynamic)0, (dynamic)1 }, new T[]{ (dynamic)1 } ),
            new TrainingDataInstance<T>(new T[]{ (dynamic)1, (dynamic)1 }, new T[]{ (dynamic)0 } ),
        };
    }
}
