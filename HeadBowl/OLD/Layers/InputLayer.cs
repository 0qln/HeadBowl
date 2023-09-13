using HeadBowl.OLD.Loss;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.OLD.Layers
{
    public class InputLayer<TFloat> : IInputLayer<TFloat>
    {
        public override int Size { get; init; }
        public override TFloat[] Values { get; set; }


        public InputLayer(
            int size)
        {
            Size = size;
            Values = new TFloat[Size];
        }
    }
}
