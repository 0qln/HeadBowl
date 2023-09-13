using HeadBowl.OLD.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.OLD.Nets
{
    public class Net<TFloat> : INet
    {
        public IInputLayer<TFloat> InputLayer { get; }
        public ILayer<TFloat>[] Layers { get; }


        public Net(IInputLayer<TFloat> inputLayer, params ILayer<TFloat>[] layers)
        {
            InputLayer = inputLayer;
            Layers = layers;
        }

        public Net(IInputLayer<TFloat> inputLayer, params ILayerBuilder<TFloat>[] layerBuilders)
        {
            InputLayer = inputLayer;

            Layers = new ILayer<TFloat>[layerBuilders.Length];
            Layers[0] = layerBuilders[0]._build(InputLayer);
            for (int i = 1; i < layerBuilders.Length; i++)
            {
                Layers[i] = layerBuilders[i]._build(Layers[i - 1]);
            }
        }


        public TFloat[] Forward(TFloat[] inputs)
        {
            InputLayer.Values = inputs;

            foreach (var layer in Layers)
            {
                layer.Forward();
            }

            return Layers[^1].Values;
        }

        public void Backward(TFloat[] inputs, TFloat[] expected)
        {
            Forward(inputs);

            TFloat[] lastGradients = Layers[^1].Backward(expected);

            for (int i = Layers.Length - 2; i >= 0; i--)
            {
                lastGradients = Layers[i].Backward(lastGradients);
            }
        }
    }
}
