using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl
{


    public interface INet<T>
    {
        public T Cost { get; }
        public void Train(double[] inputs, double[] expectedOutputs);
        public T[] Forward(double[] inputs);
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

        public void Train(double[] inputs, double[] expectedOutputs)
        {
            _net.Train(inputs, expectedOutputs);
        }

        public TPrecision[] Forward(double[] inputs)
        {
            return _net.Forward(inputs);
        }
    }


    internal class Net_64bit : INet<double>
    {
        public double[] Outputs => _layers[^1].Activations;
        public double Cost => _lastCost;

        private ILayer<double>[] _layers;
        private double _lastCost = 0;

        
        public Net_64bit(params ILayer<double>[] layers)
        {
            _layers = layers;
        }

        public void MSE(double[] outputs, double[] expected, out double result)
        {
            result = 0;
            for (int i = 0; i < outputs.Length; i++)
                result += Math.Pow(outputs[i] - expected[i], 2);
            result /= expected.Length;
        }

        public void Train(double[] inputs, double[] expectedOutputs)
        {
            Forward(inputs);
            MSE(_layers[^1].Activations, expectedOutputs, out _lastCost);

            _layers[^1].GenerateGradients(expectedOutputs);
            _layers[^1].ApplyGradients();
            for (int layer = _layers.Length-2; layer >= 0; layer--)
            {
                _layers[layer].GenerateGradients();
                _layers[layer].ApplyGradients();
            }
        }

        public double[] Forward(double[] inputs)
        {
            for (int i = 0; i < _layers[0].Size; i++)            
                _layers[0].Activations[i] = inputs[i];
            
            for (int layer =  1; layer < _layers.Length; layer++)
            {
                _layers[layer].Forward();
            }
            return _layers[^1].Activations;
        }
    }
}
