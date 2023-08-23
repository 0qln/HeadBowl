using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
{
    public interface ILayer { }

    public interface ILayer<TFloat> : ILayer
    {
        public int Size { get; init; }
        public TFloat[] Values { get; }
        public TFloat[] Forward();
        public TFloat[] Backward(TFloat[] expected);
    }
}
