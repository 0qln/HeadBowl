using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using System.Diagnostics.CodeAnalysis;

namespace HeadBowl.Loss
{
    public static class Loss
    {
        public static TFloat MSE<TFloat>(TFloat[] values, TFloat[] expected)
            where TFloat : struct
        {
            TFloat error = default;

            for (int i = 0; i < values.Length; i++)
            {
                var d = (dynamic)values[i] - (dynamic)expected[i];
                error += d * d;
            }

            return (dynamic)error / values.Length;
        }
    }
}
