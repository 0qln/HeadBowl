using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Activations
{
    public static class Activations<TFloat>
        where TFloat : IComparable<TFloat>
    {
        private static TFloat g1 => (dynamic)1;
        private static TFloat g0 => (dynamic)0;


        public static Func<TFloat, TFloat> ReLU => x => x.CompareTo(default) == -1 ? g0 : x;
        public static Func<TFloat, TFloat> ReLU_Der => x => x.CompareTo(default) == -1 ? g0 : g1;

        public static Func<TFloat, TFloat> Sigmoid => x => g1 / (g1 + MathF.Exp((dynamic)(-1) * x));
        /// <summary>When input value has already been sigmoided.</summary>
        public static Func<TFloat, TFloat> SigmoidDer => x => x * ((dynamic)g1 - x);
    }
}
