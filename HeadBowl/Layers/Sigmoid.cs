using HeadBowl.Helpers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
{
    public static class Sigmoid_64bit
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Activation(double input) => 1 / (1 + Math.Exp(-input));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ActivationDerivative(double input) => input * (1 - input); // when this is used, the input is already sigmoided.
    }

    public static class Sigmoid_32bit
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Activation(float input) => 1f / (float)(1 + Math.Exp(-input));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ActivationDerivative(float input) => input * (1f - input); // when this is used, the input is already sigmoided.
    }
}
