using HeadBowl.Helpers;

namespace HeadBowl.Layers
{
    public interface IFullyConnectedLayer<T> : ILayer<T>
    {

    }

    internal abstract class FullyConnectedLayer_64bit : IFullyConnectedLayer<double>
    {
        public int Size => _size;
        public Array Weights => _weights;
        public Array? Activations { get => _activations; set => _activations = (double[])value!; }
        public Array Gradients => _gradients;

        protected int _size;
        protected double[,] _weights; // [this, prev]
        protected double[] _biases, _activations, _lRates, _gradients;

        protected ILayer<double>? _prevLayer, _nextLayer;

        public bool IsInputLayer => _prevLayer is null;
        public bool IsOutputLayer => _nextLayer is null;

        protected double[]? _inputs;
        public Array Inputs
        {
            set
            {
                if (value.GetType() != typeof(double[]))
                {
                    throw new ArgumentException();
                }

                _inputs = value as double[];
            }
            protected get
            {
                return IsInputLayer ? 
                    _inputs ?? throw new Exception("No inputs provided.") :
                    _prevLayer!.Activations ?? throw new Exception("Previous layer has not been computed yet.");
            }
        }
        protected double[]? _expectedOutputs;
        public Array GradientDependencies
        {
            set
            {
                if (value.GetType() != typeof(double[]))
                {
                    throw new ArgumentException();
                }

                _expectedOutputs = value as double[];
            }
            protected get
            {
                if (IsOutputLayer)
                {
                    return _expectedOutputs ?? throw new Exception("No expected outputs provided.");
                }
                else
                {
                    Array ret = _nextLayer!.Gradients ?? throw new Exception("Next layers gradients have not been calculated yet.");
                    double[] specRet = ret as double[] ?? throw new NotImplementedException("Backpropagation between FC and non FC layers have not been implemented yet");
                    return specRet;
                }
            }
        }

        public bool EnableParallelProcessing { get; set; }

        abstract public double Activation(double input);
        abstract public double ActivationDerivative(double input);


        public FullyConnectedLayer_64bit(int size)
        {
            _size = size;
            _biases = Init<double>.Random(_size);
            _activations = new double[_size];
            _gradients = new double[_size];
            _lRates = Init<double>.LearningRates(_size);
            _weights = Init<double>.Random(size, 0);
        }


        public void Forward()
        {
            // TODO: we might wanna add a clean way to transform next layer arrays (which are supposed
            //       to be able to have variable dimensions) to a arrays of with the specific required
            //       dimensions.
            double[] inputs = (double[])Inputs;

            if (EnableParallelProcessing)
            {
                Parallel.For(0, _size, node =>
                {
                    _activations[node] = _biases[node];

                    for (int prevLayerNode = 0; prevLayerNode < inputs.Length; prevLayerNode++)
                        _activations[node] += _weights[node, prevLayerNode] * inputs[prevLayerNode];

                    _activations[node] = Activation(_activations[node]);
                });
            }
            else
            {
                for (int node = 0; node < _size; node++)
                {
                    _activations[node] = _biases[node];

                    for (int prevLayerNode = 0; prevLayerNode < inputs.Length; prevLayerNode++)
                        _activations[node] += _weights[node, prevLayerNode] * inputs[prevLayerNode];

                    _activations[node] = Activation(_activations[node]);
                }
            }
        }

        public void GenerateGradients()
        {
            if (IsOutputLayer)
            {
                if (EnableParallelProcessing)
                {
                    Parallel.For(0, _size, node =>
                    {
                        _gradients[node] = Math.Pow(_activations[node] - ((double[])GradientDependencies)[node], 2);
                    });
                }
                else
                {
                    for (int node = 0; node < _size; node++)
                    {
                        _gradients[node] = Math.Pow(_activations[node] - ((double[])GradientDependencies)[node], 2);
                    }
                }
            }
            else
            {
                if (EnableParallelProcessing)
                {
                    Parallel.For(0, _size, node =>
                    {
                        for (int nextLayerNode = 0; nextLayerNode < _nextLayer!.Size; nextLayerNode++)
                            // TODO: we might want to add a clean way to transform next layer arrays (which are supposed
                            //       to be able to have variable dimensions) to a arrays of with the specific required
                            //       dimensions.
                            _gradients[node] += ((double[])GradientDependencies)[nextLayerNode] * ((double[,])_nextLayer.Weights)[nextLayerNode, node];

                        _gradients[node] *= ActivationDerivative(_activations[node]);
                    });
                }
                else
                {
                    for (int node = 0; node < _size; node++)
                    {
                        for (int nextLayerNode = 0; nextLayerNode < _nextLayer!.Size; nextLayerNode++)
                            // TODO: we might want to add a clean way to transform next layer arrays (which are supposed
                            //       to be able to have variable dimensions) to a arrays of with the specific required
                            //       dimensions.
                            _gradients[node] += ((double[])GradientDependencies)[nextLayerNode] * ((double[,])_nextLayer.Weights)[nextLayerNode, node];

                        _gradients[node] *= ActivationDerivative(_activations[node]);
                    }
                }
            }
        }

        public void ApplyGradients()
        {
            if (!IsInputLayer)
            {
                if (EnableParallelProcessing)
                {
                    Parallel.For(0, _size, node =>
                    {
                        // We have to clamp the values here, other wise the output might result in NaN or an Infinity

                        _biases[node] -= Math.Clamp(_gradients[node] * _lRates[node], -1e50, 1e50);

                        for (int prevLayerNode = 0; prevLayerNode < _prevLayer!.Size; prevLayerNode++)
                            // TODO: we might wanna add a clean way to transform next layer arrays (which are supposed
                            //       to be able to have variable dimensions) to a arrays of with the specific required
                            //       dimensions.
                            //
                            // we need to clamp the value to a minimum and maximum, other wise NaNs and Inifinities can occur. This might be eliminated 
                            // later when adding optimizers with weights decay.
                            _weights[node, prevLayerNode] -= Math.Clamp(_gradients[node] * ((double[])_prevLayer.Activations)[prevLayerNode] * _lRates[node], -1e50, 1e50);
                    });
                }
                else
                {
                    for (int node = 0; node < _size; node++)
                    {
                        // We have to clamp the values here, other wise the output might result in NaN or an Infinity

                        _biases[node] -= Math.Clamp(_gradients[node] * _lRates[node], -1e50, 1e50);

                        for (int prevLayerNode = 0; prevLayerNode < _prevLayer!.Size; prevLayerNode++)
                            // TODO: we might wanna add a clean way to transform next layer arrays (which are supposed
                            //       to be able to have variable dimensions) to a arrays of with the specific required
                            //       dimensions.
                            //
                            // we need to clamp the value to a minimum and maximum, other wise NaNs and Inifinities can occur. This might be eliminated 
                            // later when adding optimizers with weights decay.
                            _weights[node, prevLayerNode] -= Math.Clamp(_gradients[node] * ((double[])_prevLayer.Activations)[prevLayerNode] * _lRates[node], -1e50, 1e50);
                    }
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



    internal class FullyConnectedSigmoidLayer_64bit : FullyConnectedLayer_64bit
    {
        public FullyConnectedSigmoidLayer_64bit(int size)
            : base(size)
        { }

        public override double Activation(double input) => Sigmoid_64bit.Activation(input);
        public override double ActivationDerivative(double input) => Sigmoid_64bit.ActivationDerivative(input);
    }

    internal class FullyConnectedReLULayer_64bit : FullyConnectedLayer_64bit
    {
        public FullyConnectedReLULayer_64bit(int size)
            : base(size)
        { }

        public override double Activation(double input) => ReLU_64bit.Activation(input);
        public override double ActivationDerivative(double input) => ReLU_64bit.ActivationDerivative(input);
    }
}
