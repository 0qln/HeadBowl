using HeadBowl.Helpers;
using System.Diagnostics;

namespace HeadBowl.Layers
{
    public enum PaddingType
    {
        ZeroPadding
    }


    public interface IConvLayer<T> : ILayer<T>
    {
        int ZeroPaddingAmount { get; }

        int Stride { get; }

        int Filters { get; } // Is equivalent to output depth
        int FilterDepth { get; } // Has to be equal to input depth
        int FilterExtend { get; } // height and width of each filter
    }

    internal class ConvReLULayer_64bit : ConvLayer_64bit
    {
        internal ConvReLULayer_64bit(int inputDepth, int outputDepth, int filterExtend, int stride, int zeroPadding) 
            : base(inputDepth, outputDepth, filterExtend, stride, zeroPadding)
        {
        }

        public override double Activation(double input) => ReLU_64bit.Activation(input);
        public override double ActivationDerivative(double input) => ReLU_64bit.ActivationDerivative(input);
    }

    internal abstract class ConvLayer_64bit : IConvLayer<double>
    {
        private double[,,,]? _gradients; // filter, extend, extend, filterDepth
        private double[,,,] _weights; // filter, extend, extend, filterDepth
        private double[] _biases; // filter
        private double[,,]? _activations; // filter, transformed input x, transformed input y

        private int _filters;
        private int _filterDepth;
        private int _filterExtend;

        private int _stride;

        private int _paddingAmount;

        private IConvLayer<double>? _prevLayer, _nextLayer;


        public int Size { get; }
        public int Filters => _filters;
        public int FilterDepth => _filterDepth;
        public int FilterExtend => _filterExtend;
        public int Stride => _stride;
        public int ZeroPaddingAmount => _paddingAmount;
        public Array? Activations { get => _activations; set => _activations = (double[,,])value!; }
        public Array? Gradients => _gradients;
        public Array Weights => _weights;

        public bool IsInputLayer => _prevLayer is null;
        public bool IsOutputLayer => _nextLayer is null;


        abstract public double Activation(double input);
        abstract public double ActivationDerivative(double input);


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


        public static void ZeroPad(in double[,,] source, double[,,] destination, int amount)
        {
            Debug.Assert(source.GetLength(0) == destination.GetLength(0));
            Debug.Assert(source.GetLength(1) == destination.GetLength(1) + amount * 2);
            Debug.Assert(source.GetLength(2) == destination.GetLength(2) + amount * 2);

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
        public void Forward(in double[]? nnInputs = null)
        {
            // dertermine the input, might be input layer or hidden (then it's the activations from the prev layer)
            double[,,] origInputs = (double[,,])(nnInputs ??
                (_prevLayer ?? throw new Exception("Prev layer not specified and no inputs provided"))
                    .Activations ?? throw new Exception("Prev layer has not been computed yet"));

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

        public void GenerateGradients(in double[]? expectedNNOutputs = null)
        {
            throw new NotImplementedException();
        }

        public void ApplyGradients()
        {
            throw new NotImplementedException();
        }

        void ILayer<double>._InitInNet(ILayer<double>? prev, ILayer<double>? next)
        {
            throw new NotImplementedException();
        }
    }

}
