using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.OLD.Activations
{
    public class ReLU<TFloat> : IActivationFunction<TFloat>
        where TFloat : IComparable<TFloat>
    {
        public TFloat Backward(TFloat x)
        {
            return x.CompareTo((dynamic)0) == -1 ? (dynamic)0 : x;
        }

        public TFloat Forward(TFloat x)
        {
            return x.CompareTo((dynamic)0) == -1 ? 0 : (dynamic)1;
        }
    }
}
