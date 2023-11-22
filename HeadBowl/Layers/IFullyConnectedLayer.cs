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



    internal class FullyConnectedLayer_64bit : IFullyConnectedLayer<double>
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
        public Array? Activations { get => _activations; set => _activations = (double[])value!; }
        public Array Gradients => _gradients;
        public Array Weights { get => _weights; set => _weights = (double[,])value; }
        public Array Biases { get => _biases; set => _biases = (double[])value; }

        protected int _size;
        protected double[,] _weights; // [this, prev]
        protected double[] _biases, _activations, _lRates, _gradients;

        protected ILayer<double>? _prevLayer, _nextLayer;

        protected IOptimizer<double> _optimizer;

        public IActivation<double> Activation => _activation;
        protected readonly IActivation<double> _activation;

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

        public Array LearningRates => _lRates;

        public IOptimizer<double> Optimizer { get => _optimizer; set => _optimizer = value; }

        public FullyConnectedLayer_64bit(int size, IActivation<double> activation, IOptimizer<double> optimizer)
        {
            _activation = activation;
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

                        activations[node] = _activation.Activation(activations[node]);
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

                        activations[node] = _activation.Activation(activations[node]);
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

                            gradients[node] *= _activation.Derivative(activations[node]);
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

                            gradients[node] *= _activation.Derivative(activations[node]);
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

        public ILayerBuilder<double> ToRawBuilder()
        {
            Type genericType = typeof(FullyConnectedLayer<,>);
            var types = new Type[] { typeof(double), Activation.ActivationType };
            Type builderType = genericType.MakeGenericType(types);
            var builder = (LayerBuilderBase<double>)Activator.CreateInstance(builderType, Size, Optimizer.Clone())!;
            return builder;
        }
    }

}
