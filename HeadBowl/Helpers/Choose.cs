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

        public static TrainingDataInstance<T> Random<T>(ITrainingData<T> data)
        {
            int index = _rng.Next(0, data.SampleCount);
            return data.Data[index];
        }

        public static TrainingDataInstance<T> First<T>(ITrainingData<T> data)
        {
            return data.Data[0];
        }

        private static int _cycleIndex = 0;
        public static TrainingDataInstance<T> Cycle<T>(ITrainingData<T> data)
        {
            return data.Data[_cycleIndex++ % data.SampleCount];
        }
    }
}
