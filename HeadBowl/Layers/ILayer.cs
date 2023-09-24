using HeadBowl.Helpers;

namespace HeadBowl.Layers
{
    public interface ILayer<T>
    {
        public int Size { get; }
        public T[,] Weights { get; }
        public T[] Activations { get; }
        public T[] Gradients { get; }
        public bool IsOutputLayer { get; }
        public bool IsInputLayer { get; }

        public void Forward(in T[]? nnInputs = null);
        public void GenerateGradients(in T[]? expectedNNOutputs = null);
        public void ApplyGradients();

        internal void _InitInNet(ILayer<T>? prev, ILayer<T>? next);
    }

    public class Layer<TPrecision>
    {
        public static ILayerBuilder<TPrecision> Create(ActivationType activation, int size)
        {
            return new LayerBuilder<TPrecision>(size, activation);
        }
    }

    public class LayerBuilder<TPrecision> : ILayerBuilder<TPrecision>
    {
        public int Size => _instance.Size;

        private readonly ILayer<TPrecision> _instance;
        private ILayer<TPrecision>? _next, _prev;


        public LayerBuilder(int size, ActivationType activation)
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
            _biases = Init<double>.Biases(_size);
            _activations = new double[_size];
            _gradients = new double[_size];
            _lRates = Init<double>.LearningRates(_size);

            _prevLayer = prevLayer;
            _nextLayer = nextLayer;
            _weights = Init<double>.Weights(size, prevLayer?.Size ?? 0);
        }
        public LayerBase_64bit(int size)
        {
            _size = size;
            _biases = Init<double>.Biases(_size);
            _activations = new double[_size];
            _gradients = new double[_size];
            _lRates = Init<double>.LearningRates(_size);
            _weights = Init<double>.Weights(size, 0);
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
            _weights = Init<double>.Weights(_size, _prevLayer?.Size ?? 0);
        }
    }
}
