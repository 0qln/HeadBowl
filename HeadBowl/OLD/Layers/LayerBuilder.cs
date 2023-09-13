using HeadBowl.OLD.Activations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.OLD.Layers
{
    public class LayerBuilder<T, TFloat> : ILayerBuilder<TFloat>
        where T : IActivationFunction<TFloat>, new()
        where TFloat : struct
    {
        private int _size;

        public LayerBuilder(int size)
        {
            _size = size;
        }

        ILayer<TFloat> ILayerBuilder<TFloat>._build(ILayer<TFloat> prevLayer)
        {
            return new Layer<T, TFloat>(prevLayer, _size);
        }
    }
}
