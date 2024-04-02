using HeadBowl.TrainingData;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Helpers
{
    public static class Choose
    {
        private static readonly Random _rng = new Random();

        public static DataInstance<T> Random<T>(ITrainingData<T> data)
        {
            int index = _rng.Next(0, data.SampleCount);
            return data.TrainingData[index];
        }

        public static DataInstance<T> Random<T>(ITestingData<T> data)
        {
            int index = _rng.Next(0, data.SampleCount);
            return data.TestingData[index];
        }

        public static DataInstance<T> First<T>(ITrainingData<T> data)
        {
            return data.TrainingData[0];
        }

        private static int _cycleIndex = 0;
        public static DataInstance<T> Cycle<T>(ITrainingData<T> data)
        {
            return data.TrainingData[_cycleIndex++ % data.SampleCount];
        }
    }
}
