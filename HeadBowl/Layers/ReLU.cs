using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Drawing;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using HeadBowl.Helpers;

namespace HeadBowl.Layers
{
    public static class ReLU_64bit
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Activation(double input) => Math.Max(input, 0);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ActivationDerivative(double input) => input < 0 ? 0 : 1;
    }
}
