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
    internal class ReLULayer_64bit : LayerBase_64bit, ILayer<double>
    {
        public ReLULayer_64bit(int size, ILayer<double>? prevLayer, ILayer<double>? nextLayer)
            : base (size, prevLayer, nextLayer)
        {
        }

        public ReLULayer_64bit(int size)
            : base (size)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override double Activation(double input) => Math.Max(input, 0);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override double ActivationDerivative(double input) => input < 0 ? 0 : 1;
    }

}
