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
    internal class SigmoidLayer_64bit : LayerBase_64bit, ILayer<double>
    {
        public SigmoidLayer_64bit(int size, ILayer<double>? prevLayer, ILayer<double>? nextLayer)
            : base(size, prevLayer, nextLayer)
        {
        }

        public SigmoidLayer_64bit(int size)
            : base(size)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override double Activation(double input) => 1 / (1 + Math.Exp(-input));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override double ActivationDerivative(double input) => input * (1 - input); // when this is used, the input is already sigmoided.
    }
}
