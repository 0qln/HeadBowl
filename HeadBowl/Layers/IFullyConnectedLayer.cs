using HeadBowl.Helpers;
using HeadBowl.Optimizers;
using Microsoft.Toolkit.HighPerformance;
using System.ComponentModel;
using System.Drawing;
using System.Runtime.CompilerServices;

namespace HeadBowl.Layers
{
    public interface IFullyConnectedLayer<T> : ILayer<T>
    {

    }



    internal abstract class FullyConnectedLayer_64bit : IFullyConnectedLayer<double>
    // Regarding the Array casting of the:
    // We might want to add a clean way to transform next layer arrays (which are supposed
    // to be able to have variable dimensions) to a arrays of with the specific required
    // dimensions.
    // Alternativly, we could insert a layer into the network as an interface between two
    // layers of different type.
    // (e.g. Getting an input from a convolutional layer)

    // Regarding the value clamping:
    // We need to clamp the value to a minimum and maximum, other wise NaNs and Inifinities can occur.
    // This might be redundant later when adding optimizers with weights decay.

    // When running in parallel, the spans have to be created for each loop as the cannot be
    // allocated on the heap. Even then, running benchmarks shows a significant advantage,
    // even for the spans that are created only to be used once.
    {
        public int Size => _size;
        public Array Weights => _weights;
        public Array? Activations { get => _activations; set => _activations = (double[])value!; }
        public Array Gradients => _gradients;

        protected int _size;
        protected double[,] _weights; // [this, prev]
        protected double[] _biases, _activations, _lRates, _gradients;

        protected ILayer<double>? _prevLayer, _nextLayer;
        protected readonly IOptimizer<double> _optimizer;

        public IOptimizer<double> Optimizer => _optimizer;

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
        public bool ExperimentalFeature { get; set; }


        abstract public double Activation(double input);
        abstract public double ActivationDerivative(double input);


        public FullyConnectedLayer_64bit(int size, IOptimizer<double> optimizer)
        {
            _optimizer = optimizer;
            _size = size;
            _biases = Init<double>.Random(_size);
            _activations = new double[_size];
            _gradients = new double[_size];
            _lRates = Init<double>.LearningRates(_size);
            _weights = Init<double>.Random(size, 0);
        }

        public void Forward()
        {
            double[] inputs = (double[])Inputs;

            if (EnableParallelProcessing)
            {
                if (ExperimentalFeature)
                {            
                }
                else
                {
                    Parallel.For(0, _size, node =>
                    {
                        Span<double> activations = _activations;
                        ReadOnlySpan<double> biases = _biases;
                        ReadOnlySpan2D<double> weights = _weights;

                        activations[node] = biases[node];
                        for (int prevLayerNode = 0; prevLayerNode < inputs.Length; prevLayerNode++)
                            activations[node] += weights[node, prevLayerNode] * inputs[prevLayerNode];

                        activations[node] = Activation(activations[node]);
                    });
                }
            }
            else
            {
                if (ExperimentalFeature)
                {
                }
                else
                {
                    Span<double> activations = _activations;
                    ReadOnlySpan<double> biases = _biases;

                    for (int node = 0; node < _size; node++)
                    {
                        activations[node] = biases[node];

                        ReadOnlySpan2D<double> weights = _weights;

                        for (int prevLayerNode = 0; prevLayerNode < inputs.Length; prevLayerNode++)
                            activations[node] += weights[node, prevLayerNode] * inputs[prevLayerNode];

                        activations[node] = Activation(activations[node]);
                    }
                }
            }
        }

        public void GenerateGradients()
        {
            if (IsOutputLayer)
            {
                if (EnableParallelProcessing)
                {
                    if (ExperimentalFeature)
                    {
                    }
                    else
                    {
                        Parallel.For(0, _size, node =>
                        {
                            Span<double> gradients =  _gradients;
                            ReadOnlySpan<double> activations = _activations, gradientDep =  (double[])GradientDependencies;
                            gradients[node] = Math.Pow(activations[node] - gradientDep[node], 2);
                        });
                    }
                }
                else
                {
                    if (ExperimentalFeature)
                    {
                    }
                    else
                    {
                        Span<double> gradients = _gradients;
                        ReadOnlySpan<double> activations = _activations, gradientDep = (double[])GradientDependencies;
                        for (int node = 0; node < _size; node++)
                        {
                            gradients[node] = Math.Pow(activations[node] - gradientDep[node], 2);
                        }
                    }
                }
            }
            else
            {
                if (EnableParallelProcessing)
                {
                    if (ExperimentalFeature)
                    {
                    }
                    else
                    {
                        Parallel.For(0, _size, node =>
                        {
                            ReadOnlySpan2D<double> nextLayer = (double[,])_nextLayer!.Weights;
                            ReadOnlySpan<double> gradientDep = (double[])GradientDependencies, activations = _activations;
                            Span<double> gradients = _gradients;

                            for (int nextLayerNode = 0; nextLayerNode < _nextLayer!.Size; nextLayerNode++)
                                gradients[node] += gradientDep[nextLayerNode] * nextLayer[nextLayerNode, node];

                            gradients[node] *= ActivationDerivative(activations[node]);
                        });
                    }
                }
                else
                {
                    if (ExperimentalFeature)
                    {
                    }
                    else
                    {
                        ReadOnlySpan2D<double> nextLayer = (double[,])_nextLayer!.Weights; // suppress warning: if not output layer, must have next layer 
                        ReadOnlySpan<double> gradientDep = (double[])GradientDependencies, activations = _activations;
                        Span<double> gradients = _gradients;

                        for (int node = 0; node < _size; node++)
                        {
                            for (int nextLayerNode = 0; nextLayerNode < _nextLayer!.Size; nextLayerNode++)
                                gradients[node] += gradientDep[nextLayerNode] * nextLayer[nextLayerNode, node];

                            gradients[node] *= ActivationDerivative(activations[node]);
                        }
                    }
                }
            }
        }

        public void ApplyOptimizer()
        {
            _optimizer.Optimize(this);
        }

        public void ApplyGradients()
        {
            if (!IsInputLayer)
            {
                if (EnableParallelProcessing)
                {
                    if (ExperimentalFeature)
                    {
                    }
                    else
                    {
                        Parallel.For(0, _size, node =>
                        {
                            ReadOnlySpan<double> gradients = _gradients, lRates = _lRates, prevLayer = (double[])_prevLayer!.Activations!;
                            Span<double> biases = _biases;
                            Span2D<double> weights = _weights;

                            biases[node] -= Math.Clamp(gradients[node] * lRates[node], -1e50, 1e50);

                            for (int prevLayerNode = 0; prevLayerNode < _prevLayer!.Size; prevLayerNode++)                                
                                weights[node, prevLayerNode] -= Math.Clamp(gradients[node] * prevLayer[prevLayerNode] * lRates[node], -1e50, 1e50);
                        });
                    }
                }
                else
                {
                    if (ExperimentalFeature)
                    {
                    }
                    else
                    {
                        ReadOnlySpan<double> gradients = _gradients, lRates = _lRates, prevLayer = (double[])_prevLayer!.Activations!;
                        Span<double> biases = _biases;
                        Span2D<double> weights = _weights;

                        for (int node = 0; node < _size; node++)
                        {
                            biases[node] -= Math.Clamp(gradients[node] * lRates[node], -1e50, 1e50);

                            for (int prevLayerNode = 0; prevLayerNode < _prevLayer!.Size; prevLayerNode++)
                                weights[node, prevLayerNode] -= Math.Clamp(gradients[node] * prevLayer[prevLayerNode] * lRates[node], -1e50, 1e50);
                        }
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
        public FullyConnectedSigmoidLayer_64bit(int size, IOptimizer<double> optimizer)
            : base(size, optimizer)
        { }

        public override double Activation(double input) => Sigmoid_64bit.Activation(input);
        public override double ActivationDerivative(double input) => Sigmoid_64bit.ActivationDerivative(input);
    }


    internal class FullyConnectedReLULayer_64bit : FullyConnectedLayer_64bit
    {
        public FullyConnectedReLULayer_64bit(int size, IOptimizer<double> optimizer)
            : base(size, optimizer)
        { }

        public override double Activation(double input) => ReLU_64bit.Activation(input);
        public override double ActivationDerivative(double input) => ReLU_64bit.ActivationDerivative(input);
    }



    internal abstract class FullyConnectedLayer_32bit : IFullyConnectedLayer<float>
    {
        protected readonly IOptimizer<float> _optimizer;
        protected readonly int _size;
        protected readonly float[] _biases, _activations, _gradients, _lRates;
        protected readonly float[,] _weights;

        public IOptimizer<float> Optimizer => throw new NotImplementedException();

        public bool IsOutputLayer => throw new NotImplementedException();

        public bool IsInputLayer => throw new NotImplementedException();

        public bool EnableParallelProcessing { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public bool ExperimentalFeature { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public int Size => throw new NotImplementedException();

        public Array Inputs { set => throw new NotImplementedException(); }
        public Array GradientDependencies { set => throw new NotImplementedException(); }
        public Array? Activations { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public Array? Gradients => throw new NotImplementedException();

        public Array Weights => throw new NotImplementedException();

        IOptimizer<float> ILayer<float>.Optimizer => throw new NotImplementedException();

        bool ILayer<float>.IsOutputLayer => throw new NotImplementedException();

        bool ILayer<float>.IsInputLayer => throw new NotImplementedException();

        bool ILayer<float>.EnableParallelProcessing { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        bool ILayer<float>.ExperimentalFeature { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        int ILayer<float>.Size => throw new NotImplementedException();

        Array ILayer<float>.Inputs { set => throw new NotImplementedException(); }
        Array ILayer<float>.GradientDependencies { set => throw new NotImplementedException(); }
        Array? ILayer<float>.Activations { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        Array? ILayer<float>.Gradients => throw new NotImplementedException();

        Array ILayer<float>.Weights => throw new NotImplementedException();


        public FullyConnectedLayer_32bit(int size, IOptimizer<float> optimizer)
        {
            _optimizer = optimizer;
            _size = size;
            _biases = Init<float>.Random(_size);
            _activations = new float[_size];
            _gradients = new float[_size];
            _lRates = Init<float>.LearningRates(_size);
            _weights = Init<float>.Random(size, 0);
        }


        abstract public float Activation(float input);
        abstract public float ActivationDerivative(float input);

        public void ApplyGradients()
        {
            throw new NotImplementedException();
        }

        public void ApplyOptimizer()
        {
            throw new NotImplementedException();
        }

        public void Forward()
        {
            throw new NotImplementedException();
        }

        public void GenerateGradients()
        {
            throw new NotImplementedException();
        }

        void ILayer<float>.ApplyGradients()
        {
            throw new NotImplementedException();
        }

        void ILayer<float>.ApplyOptimizer()
        {
            throw new NotImplementedException();
        }

        void ILayer<float>.Forward()
        {
            throw new NotImplementedException();
        }

        void ILayer<float>.GenerateGradients()
        {
            throw new NotImplementedException();
        }

        void ILayer<float>._InitInNet(ILayer<float>? prev, ILayer<float>? next)
        {
            throw new NotImplementedException();
        }
    }



    internal class FullyConnectedSigmoidLayer_32bit : FullyConnectedLayer_32bit
    {
        public FullyConnectedSigmoidLayer_32bit(int size, IOptimizer<float> optimizer)
            : base(size, optimizer)
        { }


        public override float Activation(float input) => Sigmoid_32bit.Activation(input);
        public override float ActivationDerivative(float input) => Sigmoid_32bit.ActivationDerivative(input);
    }
}
