using HeadBowl.Loss;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
{
    public class InputLayer<TFloat> : IInputLayer<TFloat>
    {
        public int Size { get; internal set; }
        public TFloat[] Values { get; set; }


        public InputLayer(
            int size)
        {
            Size = size;
            Values = new TFloat[Size];
        }

        public IInitializable Init(params object[] parameters)
        {
            throw new NotImplementedException();
        }
    }
}
