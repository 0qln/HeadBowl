using HeadBowl.OLD.Activations;
using HeadBowl.OLD.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl
{
    public interface ILayer<T>
    {
        public int Size { get; }
        public T[,] Weights { get; }
        public T[] Activations { get; }
        public T[] Gradients { get; }

        public void Forward();
        public void ResetGradients();
        public void GenerateGradients();
        public void ApplyGradients();
    }


    internal static class Init<T>
    {
        private static System.Random rng = new();

        public static T[,] Weights(int size1, int size2)
        {
            var result = new T[size1, size2];

            for (int i = 0; i < size1; i++)
            {
                for (int j =  0; j < size2; j++)
                {
                    result[i, j] = (dynamic)rng.NextDouble();
                }
            }

            return result;
        }

        public static T[] Biases(int size)
        {
            var result = new T[size];

            for (int i = 0; i < size; i++)
            {
                result[i] = (dynamic)rng.NextDouble();
            }

            return result;
        }

        public static T[] LearningRates(int size)
        {
            var result = new T[size];

            for (int i = 0; i < size; i++)
            {
                result[i] = (dynamic)0.0001;
            }

            return result;
        }
    }


    public class ReLULayer<TPrecision> : ILayer<TPrecision>
    {
        public int Size => _layer.Size;
        public TPrecision[,] Weights => _layer.Weights;
        public TPrecision[] Activations => _layer.Activations;
        public TPrecision[] Gradients => _layer.Gradients;

        private ILayer<TPrecision> _layer;


        public ReLULayer(
            ILayer<TPrecision> prevLayer,
            ILayer<TPrecision> nextLayer,
            int size)
        {
            _layer =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_16bit((ILayer<Half>)prevLayer, (ILayer<Half>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_32bit((ILayer<Single>)prevLayer, (ILayer<Single>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new ReLULayer_64bit((ILayer<Double>)prevLayer, (ILayer<Double>)nextLayer, size) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_128bit((ILayer<Decimal>)prevLayer, (ILayer<Decimal>)nextLayer, size)*/
                : throw new NotImplementedException();
        }

        public void Forward() => _layer.Forward();
        public void ResetGradients() => _layer.ResetGradients();
        public void GenerateGradients() => _layer.GenerateGradients();
        public void ApplyGradients() => _layer.ApplyGradients();
    }


    internal class ReLULayer_64bit : ILayer<Double>
    {
        public int Size => _size;
        public double[,] Weights => _weights;
        public double[] Activations => _activations;
        public double[] Gradients => _gradients;

        private int _size;
        private double[,] _weights; // [this, prev]
        private double[] _biases, _activations, _lRates, _gradients;

        private ILayer<Double> _prevLayer, _nextLayer;


        public ReLULayer_64bit(
            ILayer<Double> prevLayer,
            ILayer<Double> nextLayer, 
            int size)
        {
            _weights = Init<double>.Weights(size, prevLayer.Size);
            _biases = Init<double>.Biases(size);
            _activations = new double[size];
            _gradients = new double[size];
            _lRates = Init<double>.LearningRates(size);
            _prevLayer = prevLayer;
            _nextLayer = nextLayer;
            _size = size;
        }

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Activation(double input) => Math.Max(input, 0);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ActivationDerivative(double input) => input < 0 ? 0 : 1;


        public void Forward()
        {
            for (int node = 0; node < _size; node++)
            {
                _activations[node] = _biases[node];

                for (int prevLayerNode = 0; prevLayerNode < _prevLayer.Size; prevLayerNode++)
                    _activations[node] += _weights[node, prevLayerNode] * _prevLayer.Activations[prevLayerNode];
                
                _activations[node] = Activation(_activations[node]);
            }
        }

        public void ResetGradients()
        {
            _gradients = new double[_size];
        }

        public void GenerateGradients()
        {
            for (int node = 0; node < _size; node++)
            {
                for (int nextLayerNode = 0; nextLayerNode < _nextLayer.Size; nextLayerNode++)                
                    _gradients[node] += _nextLayer.Gradients[nextLayerNode] * _nextLayer.Weights[nextLayerNode, node];
                
                _gradients[node] *= ActivationDerivative(_activations[node]);
            }
        }

        public void ApplyGradients()
        {

            for (int node = 0; node < _size; node++)
            {
                _biases[node] -= _nextLayer.Gradients[node] * _lRates[node];

                for (int prevLayerNode = 0; prevLayerNode < _prevLayer.Size; prevLayerNode++)
                    _weights[node, prevLayerNode] -= _gradients[node] * _prevLayer.Activations[prevLayerNode] * _lRates[node];
                
            }
        }
    }

}
