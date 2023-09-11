using HeadBowl.Activations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
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
            return new HiddenLayer<T, TFloat>(prevLayer, _size);
        }

        internal ILayer<TFloat> _build<TLayer>(ILayer<TFloat> prevLayer)
            where TLayer : ILayer<TFloat>
        {
            // Create an instance of TLayer using reflection
            var layerInstance = Activator.CreateInstance(typeof(TLayer)) as ILayer<TFloat>;

            if (layerInstance is null) throw new ArgumentException();

            // Initialize the instance with prevLayer and _size
            layerInstance.Init(prevLayer, _size);

            // Return the created instance
            return layerInstance;
        }
    }
}
