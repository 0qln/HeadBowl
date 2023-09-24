using HeadBowl.OLD.Activations;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Drawing;
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

        public void Forward(in T[]? nnInputs = null);
        public void GenerateGradients(in T[]? expectedNNOutputs = null);
        public void ApplyGradients();

        internal void _InitInNet(ILayer<T>? prev, ILayer<T>? next);
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
                result[i] = (dynamic)0.00001;
            }

            return result;
        }
    }

    public interface ILayerBuilder<T>
    {
        public int Size { get; }
        internal void SetNext(ILayer<T>? layer);
        internal void SetPrev(ILayer<T>? layer);
        public ILayer<T> Build();
        public ILayer<T> Instance();
    }


    public class ReLULayerBuilder<TPrecision> : ILayerBuilder<TPrecision>
    {
        public int Size => _instance.Size;

        int ILayerBuilder<TPrecision>.Size => throw new NotImplementedException();

        private readonly ILayer<TPrecision> _instance;
        private ILayer<TPrecision>? _next, _prev;



        public ReLULayerBuilder(int size)
        {
            _instance = new ReLULayer<TPrecision>(size);
        }


        void ILayerBuilder<TPrecision>.SetNext(ILayer<TPrecision>? layer)
        {
            _next = layer;
        }

        void ILayerBuilder<TPrecision>.SetPrev(ILayer<TPrecision>? layer)
        {
            _prev = layer;
        }

        ILayer<TPrecision> ILayerBuilder<TPrecision>.Build()
        {
            _instance._InitInNet(_prev, _next);
            return _instance;
        }

        ILayer<TPrecision> ILayerBuilder<TPrecision>.Instance()
        {
            return _instance;
        }
    }

    public class ReLULayer<TPrecision> : ILayer<TPrecision>
    {
        public int Size => _layer.Size;
        public TPrecision[,] Weights => _layer.Weights;
        public TPrecision[] Activations { get => _layer.Activations; }
        public TPrecision[] Gradients => _layer.Gradients;

        private ILayer<TPrecision> _layer;


        public ReLULayer(int size)
        {
            _layer =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_16bit((ILayer<Half>)prevLayer, (ILayer<Half>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_32bit((ILayer<Single>)prevLayer, (ILayer<Single>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new ReLULayer_64bit(size) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_128bit((ILayer<Decimal>)prevLayer, (ILayer<Decimal>)nextLayer, size)*/
                : throw new NotImplementedException();
        }

        public ReLULayer(
            ILayer<TPrecision> prevLayer,
            ILayer<TPrecision> nextLayer,
            int size)
        {
            _layer =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_16bit((ILayer<Half>)prevLayer, (ILayer<Half>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_32bit((ILayer<Single>)prevLayer, (ILayer<Single>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new ReLULayer_64bit(size, (ILayer<Double>)prevLayer, (ILayer<Double>)nextLayer) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_128bit((ILayer<Decimal>)prevLayer, (ILayer<Decimal>)nextLayer, size)*/
                : throw new NotImplementedException();
        }

        public void Forward(in TPrecision[]? inputs) => _layer.Forward(inputs);
        public void GenerateGradients(in TPrecision[]? expectedNNOutputs = null) => _layer.GenerateGradients(expectedNNOutputs);
        public void ApplyGradients() => _layer.ApplyGradients();


        void ILayer<TPrecision>._InitInNet(ILayer<TPrecision>? prev, ILayer<TPrecision>? next)
        {
            _layer._InitInNet(prev, next);
        }
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

        private ILayer<Double>? _prevLayer, _nextLayer;

        public bool IsInputLayer() => _prevLayer is null;
        public bool IsOutputLayer() => _nextLayer is null;


        public ReLULayer_64bit(
            int size,
            ILayer<Double> prevLayer,
            ILayer<Double> nextLayer)
        {
            _size = size;
            InitSizeDependencies();

            _prevLayer = prevLayer;
            _nextLayer = nextLayer;
            _weights = Init<double>.Weights(size, prevLayer.Size);
        }

        public ReLULayer_64bit(
            int size)
        {
            _size = size;
            InitSizeDependencies();
        }

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Activation(double input) => Math.Max(input, 0);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ActivationDerivative(double input) => input < 0 ? 0 : 1;


        public void Forward(in double[]? nnInputs = null)
        {
            double[] inputs = IsInputLayer() ? nnInputs! : _prevLayer!.Activations;

            for (int node = 0; node < _size; node++)
            {
                _activations[node] = _biases[node];

                for (int prevLayerNode = 0; prevLayerNode < inputs.Length; prevLayerNode++)
                    _activations[node] += _weights[node, prevLayerNode] * inputs[prevLayerNode]; //_weights breaks here

                _activations[node] = Activation(_activations[node]);
            }
        }

        public void GenerateGradients(in double[]? expectedNNOutputs = null)
        {
            if (IsOutputLayer())
            {
                for (int node = 0; node < _size; node++)
                {
                    _gradients[node] = Math.Pow(_activations[node] - expectedNNOutputs![node], 2);
                }
            }
            else
            {
                for (int node = 0; node < _size; node++)
                {
                    for (int nextLayerNode = 0; nextLayerNode < _nextLayer!.Size; nextLayerNode++)                
                        _gradients[node] += _nextLayer.Gradients[nextLayerNode] * _nextLayer.Weights[nextLayerNode, node];
                
                    _gradients[node] *= ActivationDerivative(_activations[node]);
                }
            }
        }

        public void ApplyGradients()
        {
            if (!IsInputLayer())
            {
                for (int node = 0; node < _size; node++)
                {
                    _biases[node] -= Math.Clamp(_gradients[node] * _lRates[node], -1e50, 1e50);

                    for (int prevLayerNode = 0; prevLayerNode < _prevLayer!.Size; prevLayerNode++)
                        _weights[node, prevLayerNode] -= Math.Clamp(_gradients[node] * _prevLayer.Activations[prevLayerNode] * _lRates[node], -1e50, 1e50);

                }
            }
        }

        void ILayer<double>._InitInNet(ILayer<double>? prev, ILayer<double>? next)
        {
            _prevLayer = prev;
            _nextLayer = next;
            _weights = Init<double>.Weights(_size, _prevLayer?.Size ?? 0);
        }

        private void InitSizeDependencies()
        {
            _biases = Init<double>.Biases(_size);
            _activations = new double[_size];
            _gradients = new double[_size];
            _lRates = Init<double>.LearningRates(_size);
        }
    }

}
