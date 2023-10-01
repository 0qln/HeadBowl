using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HeadBowl.Layers;

namespace HeadBowl
{
    public interface INet<T>
    {
        public T Cost { get; }
        public void Train(T[] inputs, T[] expectedOutputs);
        public T[] Forward(Array inputs);
    }



    public class Net<TPrecision> : INet<TPrecision>
    {
        public TPrecision Cost => _net.Cost;

        private INet<TPrecision> _net;


        internal Net(params ILayer<TPrecision>[] layers)
        {            
            _net =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() :
                typeof(TPrecision) == typeof(double) ? (INet<TPrecision>)new Net_64bit((ILayer<double>[])layers) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() 
                : throw new NotImplementedException();
        }

        public static Net<TPrecision> Build(params ILayerBuilder<TPrecision>[] layerBuilders)
        {
            var layers = new List<ILayer<TPrecision>>();

            layerBuilders[ 0].SetNext(layerBuilders[ 1].Instance());
            layerBuilders[^1].SetPrev(layerBuilders[^2].Instance());
            for (int layer = 1; layer < layerBuilders.Length-1; layer++)
            {
                layerBuilders[layer].SetPrev(layerBuilders[layer - 1].Instance());
                layerBuilders[layer].SetNext(layerBuilders[layer + 1].Instance());
            }

            foreach (var layer in layerBuilders)
            {
                layers.Add(layer.Build());
            }

            return new Net<TPrecision>(layers.ToArray());
        }

        public void Train(TPrecision[] inputs, TPrecision[] expectedOutputs)
        {
            _net.Train(inputs, expectedOutputs);
        }

        public TPrecision[] Forward(Array inputs)
        {
            return _net.Forward(inputs);
        }
    }


    internal class Net_64bit : INet<double>
    {
        public Array? Outputs => _layers[^1].Activations;
        public double Cost => _lastCost;

        private ILayer<double>[] _layers;
        private double _lastCost = 0;

        
        public Net_64bit(params ILayer<double>[] layers)
        {
            _layers = layers;
        }

        public static void MSE(double[] outputs, double[] expected, out double result)
        {
            result = 0;
            for (int i = 0; i < outputs.Length; i++)
                result += Math.Pow(outputs[i] - expected[i], 2);
            result /= expected.Length;
        }

        public void Train(double[] inputs, double[] expectedOutputs)
        {
            Forward(inputs);

            MSE(((double[])_layers[^1].Activations!) 
                    ?? throw new Exception("Last layer has to have an onedimensional array as output. Consider rethinking your network design."), 
                expectedOutputs, 
                out _lastCost);

            _layers[^1].GradientDependencies = expectedOutputs;
            _layers[^1].GenerateGradients();
            _layers[^1].ApplyGradients();
            for (int layer = _layers.Length-2; layer >= 0; layer--)
            {
                _layers[layer].GenerateGradients();
                _layers[layer].ApplyGradients();
            }
        }

        public double[] Forward(Array inputs)
        {
            _layers[0].Activations = inputs;
            
            for (int layer =  1; layer < _layers.Length; layer++)
            {
                _layers[layer].Forward();
            }

            return ((double[])_layers[^1].Activations!) 
                ?? throw new Exception("Last layer has to have an onedimensional array as output. Consider rethinking your network design.");
        }
    }
}
