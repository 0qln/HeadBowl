using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.OLD.Layers
{
    public abstract class IInputLayer<TFloat> : ILayer<TFloat>
    {
        public abstract int Size { get; init; }
        public abstract TFloat[] Values { get; set; }

        // Dummies
        public TFloat[] Forward() { return null!; }
        public TFloat[] Backward(TFloat[] expected) { return null!; }
    }
}
