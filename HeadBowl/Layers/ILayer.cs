using HeadBowl.Helpers;
using System.Diagnostics;
using System.IO.Pipes;

namespace HeadBowl.Layers
{
    public interface IPoolLayer<T>
    {
    }

    public enum PaddingType
    {
        ZeroPadding
    }
    
    public interface IConvLayer<T>
    {
        public bool IsOutputLayer { get; }
        public bool IsInputLayer { get; }

        public int ZeroPaddingAmount { get; }

        public int Stride { get; }

        public int Filters { get; } // Is equivalent to output depth
        public int FilterDepth { get; } // Has to be equal to input depth
        public int FilterExtend { get; } // height and width of each filter

        public T[,,]? FeatureMaps { get; }

        public void Forward(in T[,,]? nnInputs = null);
        public void GenerateGradients(in T[,,]? expectedNNOutputs = null);
        public void ApplyGradients();
    }

    internal abstract class ConvLayer_64bit : IConvLayer<double>
    {

        private double[,,] _gradients; // filter, , filterDepth ??
        private double[,,,] _weights; // filter, extend, extend, filterDepth
        private double[] _biases; // filter
        private double[,,] _activations; // filter, transformed input x, transformed input y

        private int _filters;
        private int _filterDepth;
        private int _filterExtend;

        private int _stride;

        private int _paddingAmount;


        public int Filters => _filters;
        public int FilterDepth => _filterDepth;
        public int FilterExtend => _filterExtend;

        public int Stride => _stride;

        public int ZeroPaddingAmount => _paddingAmount;

        public bool IsInputLayer => _prevLayer is null;
        public bool IsOutputLayer => _nextLayer is null;

        public double[,,] FeatureMaps => _activations;

        private IConvLayer<double>? _prevLayer, _nextLayer;



        public ConvLayer_64bit(int inputDepth, int outputDepth, int filterExtend, int stride, int zeroPadding)
        {
            _filters = outputDepth;
            _filterDepth = inputDepth;
            _filterExtend = filterExtend;
            _stride = stride;
            _paddingAmount = zeroPadding;

            _weights = Init<double>.Random(_filters, _filterExtend, _filterExtend, _filterDepth);
            _biases = Init<double>.Random(_filters);
        }

        abstract public double Activation(double input);
        abstract public double ActivationDerivative(double input);

        public static void ZeroPad(in double[,,] source, double[,,] destination, int amount)
        {
            Debug.Assert(source.GetLength(0) == destination.GetLength(0));
            Debug.Assert(source.GetLength(1) == destination.GetLength(1) + amount*2);
            Debug.Assert(source.GetLength(2) == destination.GetLength(2) + amount*2);

            for (int d = 0; d < source.GetLength(0); d++)
            for (int x = 0; x < source.GetLength(1); x++)
            for (int y = 0; y < source.GetLength(2); y++)
            {
                destination[d, x + amount, y + amount] = source[d, x, y];
            }
        }

        /// <summary></summary>
        /// <param name="nnInputs">
        /// Dimension 0: Channels
        /// Dimension 1: Width
        /// Dimension 2: Height
        /// </param>
        /// <exception cref="ArgumentNullException"></exception>
        public void Forward(in double[,,]? nnInputs = null)
        {
            // dertermine the input, might be input layer or hidden (then it's the activations from the prev layer)
            double[,,] origInputs = nnInputs ?? 
                (_prevLayer ?? throw new Exception("Prev layer not specified and no inputs provided"))
                .FeatureMaps ?? throw new Exception("Prev layer has not been computed yet");

            // init inputs with zero padding
            var inputs = new double[_filterDepth, origInputs.GetLength(1) + ZeroPaddingAmount, origInputs.GetLength(2) + ZeroPaddingAmount];
            ZeroPad(origInputs, inputs, ZeroPaddingAmount);

            // init outputs
            _activations = new double[
                _filters,
                inputs.GetLength(1) - _filterExtend / 2 + _paddingAmount,
                inputs.GetLength(2) - _filterExtend / 2 + _paddingAmount];

            // iterate filters
            Debug.Assert(_activations.GetLength(0) == _filters);
            Debug.Assert(_weights.GetLength(0) == _filters);
            Debug.Assert(_weights.GetLength(2) == _filterDepth);

            for (int outputDepth = 0; outputDepth < _filters; outputDepth++)
            for (int outputX = 0; outputX < _activations.GetLength(1); outputX++)
            for (int outputY = 0; outputY < _activations.GetLength(2); outputY++)
            {
                _activations[outputDepth, outputX, outputY] = _biases[outputDepth];

                for (int inputDepth = 0; inputDepth < _filterDepth; inputDepth++)
                for (int weightX = 0; weightX < _filterExtend; weightX++)
                for (int weightY = 0; weightY < _filterExtend; weightY++)
                {
                    _activations[outputDepth, outputX, outputY] += _weights[outputDepth, weightX, weightY, inputDepth] * inputs[inputDepth, outputX + weightX - ZeroPaddingAmount, outputY + weightY - ZeroPaddingAmount];
                }

                _activations[outputDepth, outputX, outputY] = Activation(_activations[outputDepth, outputX, outputY]);
            }            
        }

        public void GenerateGradients(in double[,,]? expectedNNOutputs = null)
        {
            _gradients = new double[_filters, _filterExtend, _filterExtend, _filterDepth];

            if (IsOutputLayer)
            {
                for (int outputDepth = 0; outputDepth < _filters; outputDepth++)
                for (int outputX = 0; outputX < _activations.GetLength(1); outputX++)
                for (int outputY = 0; outputY < _activations.GetLength(2); outputY++)
                {
                    _gradients
                }
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
            throw new NotImplementedException();
        }

    }

    public interface IRecurrentLayer<T>
    {
    }

    public interface INormalizationLayer<T>
    {
    }

    public interface ILayer<T>
    {
        // Properties
        public bool IsOutputLayer { get; }
        public bool IsInputLayer { get; }

        // Size
        public int Size { get; }
        public T[,] Weights { get; }
        public T[] Activations { get; }
        public T[] Gradients { get; }

        public void Forward(in T[]? nnInputs = null);
        public void GenerateGradients(in T[]? expectedNNOutputs = null);
        public void ApplyGradients();

        internal void _InitInNet(ILayer<T>? prev, ILayer<T>? next);
    }


    public class Layer<TPrecision> : ILayerBuilder<TPrecision>
    {
        public int Size => _instance.Size;

        private readonly ILayer<TPrecision> _instance;
        private ILayer<TPrecision>? _next, _prev;


        private Layer(int size, ActivationType activation)
        {
            _instance = activation switch
            {
                ActivationType.Sigmoid =>
                    typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new SigmoidLayer_64bit(size)
                    : throw new NotImplementedException(),

                ActivationType.ReLU =>
                    typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new ReLULayer_64bit(size)
                    : throw new NotImplementedException(),

                _ => throw new NotImplementedException()
            };
        }
        public static Layer<TPrecision> Create(ActivationType activation, int size)
        {
            return new Layer<TPrecision>(size, activation);
        }

        void ILayerBuilder<TPrecision>.SetNext(ILayer<TPrecision>? layer) => _next = layer;
        void ILayerBuilder<TPrecision>.SetPrev(ILayer<TPrecision>? layer) => _prev = layer;

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


    internal abstract class LayerBase_64bit
    {
        public int Size => _size;
        public double[,] Weights => _weights;
        public double[] Activations => _activations;
        public double[] Gradients => _gradients;

        protected int _size;
        protected double[,] _weights; // [this, prev]
        protected double[] _biases, _activations, _lRates, _gradients;

        protected ILayer<double>? _prevLayer, _nextLayer;

        public bool IsInputLayer => _prevLayer is null;
        public bool IsOutputLayer => _nextLayer is null;


        abstract public double Activation(double input);
        abstract public double ActivationDerivative(double input);


        public LayerBase_64bit(int size, ILayer<double>? prevLayer, ILayer<double>? nextLayer)
        {
            _size = size;
            _biases = Init<double>.Random(_size);
            _activations = new double[_size];
            _gradients = new double[_size];
            _lRates = Init<double>.LearningRates(_size);

            _prevLayer = prevLayer;
            _nextLayer = nextLayer;
            _weights = Init<double>.Random(size, prevLayer?.Size ?? 0);
        }
        public LayerBase_64bit(int size)
        {
            _size = size;
            _biases = Init<double>.Random(_size);
            _activations = new double[_size];
            _gradients = new double[_size];
            _lRates = Init<double>.LearningRates(_size);
            _weights = Init<double>.Random(size, 0);
        }


        public void Forward(in double[]? nnInputs = null)
        {
            double[] inputs = IsInputLayer ? nnInputs! : _prevLayer!.Activations;

            for (int node = 0; node < _size; node++)
            {
                _activations[node] = _biases[node];

                for (int prevLayerNode = 0; prevLayerNode < inputs.Length; prevLayerNode++)
                    _activations[node] += _weights[node, prevLayerNode] * inputs[prevLayerNode];

                _activations[node] = Activation(_activations[node]);
            }
        }

        public void GenerateGradients(in double[]? expectedNNOutputs = null)
        {
            if (IsOutputLayer)
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
            if (!IsInputLayer)
            {
                for (int node = 0; node < _size; node++)
                {
                    // We have to clamp the values here, other wise the might result in NaN or Infinity

                    _biases[node] -= Math.Clamp(_gradients[node] * _lRates[node], -1e50, 1e50);

                    for (int prevLayerNode = 0; prevLayerNode < _prevLayer!.Size; prevLayerNode++)
                        // we need to clamp the value to a minimum and maximum, other wise NaNs and Inifinities can occur. This might be eliminated 
                        // later when adding optimizers with weights decay.
                        _weights[node, prevLayerNode] -= Math.Clamp(_gradients[node] * _prevLayer.Activations[prevLayerNode] * _lRates[node], -1e50, 1e50);
                }
            }
        }

        public void _InitInNet(ILayer<double>? prev, ILayer<double>? next)
        {
            _prevLayer = prev;
            _nextLayer = next;
            _weights = Init<double>.Random(_size, _prevLayer?.Size ?? 0);
        }
    }
}
