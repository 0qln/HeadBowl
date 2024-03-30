using HeadBowl.Training_Data;
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
            int index = _rng.Next(0, data.Samples);
            return data.Data[index];
        }
    }
}
