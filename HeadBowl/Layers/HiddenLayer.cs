using System.Drawing;
using System.Numerics;
using System.Reflection.Emit;
using HeadBowl.Activations;
using HeadBowl.Loss;

namespace HeadBowl.Layers
{
    /*
    public class HiddenLayer<TActivation, TFloat> : IComputationalLayer<TFloat>
        where TFloat : struct
        where TActivation : IActivationFunction<TFloat>, new()
    {
        public ILayer<TFloat> PrevLayer { get; private set; }
        public IComputationalLayer<TFloat> NextLayer { get; private set; }
        public int Size { get; private set; }
        public TFloat[] Biases { get; private set; }
        public TFloat[,] Weights { get; private set; }
        public TFloat[] Values { get; private set; }

        // Learning
        public TFloat LastGradients { get; private set; }
        public TFloat LearningRate { get; set; }
        public TFloat Cost { get; private set; }

        private readonly TActivation _activate = new();


        public HiddenLayer(
            ILayer<TFloat> prevLayer, 
            int size)
        {
            PrevLayer = prevLayer;
            Size = size;
            Biases = new TFloat[Size];
            Weights = new TFloat[Size, PrevLayer.Size];
            Values = new TFloat[Size];
        }

#pragma warning disable CS8618 // This is used by the `Activator.CreateInstance` function
        internal HiddenLayer() { }
#pragma warning restore CS8618 

        IInitializable IInitializable.Init(params object[] parameters)
        {
            PrevLayer = parameters[0] as ILayer<TFloat> ?? throw new ArgumentException();
            Size = (int)parameters[1];
            Biases = new TFloat[Size];
            Weights = new TFloat[Size, PrevLayer.Size];
            Values = new TFloat[Size];

            return this;
        }


        public TFloat[] Forward()
        {
            var result = new TFloat[Size];

            for (int node = 0; node < Size; node++)
            {
                result[node] = Biases[node];

                for (int weight = 0; weight < PrevLayer.Size; weight++)
                {
                    result[node] += (dynamic)Weights[node, weight] * (dynamic)PrevLayer.Values[weight];
                }

                result[node] = _activate.Forward(result[node]);
            }

            return result;
        }


        /// <summary>
        /// The backpropagation funtion for the hidden layers.
        /// </summary>
        /// <param name="expected"></param>
        /// <returns>The gradients for the previous layer</returns>
        public TFloat[] Backward()
        {
            var gradients = new TFloat[Size];

            // calculate gradients
            for (int node = 0; node < Size; node++)
            {
                for (int nextLayerNode = 0; nextLayerNode < NextLayer.Size; nextLayerNode++)
                {
                    gradients[node] += NextLayer.LastGradients[nextLayerNode] *  Weights[nextLayerNode, node];
                }
                gradients[node] *= (dynamic)_activate.Backward(Values[node]);
            }

            //iterate over outputs of layer
            for (int node = 0; node < Size; node++)
            {
                // modify biases of network
                _biases[layer - 1][node] -= gradients[layer][node] * LearningRate;
                // iterate over inputs to layer
                for (int weight = 0; weight < _layers[layer - 1]; weight++)
                {
                    // modify weights of network
                    _weights[layer - 1][node][weight] -= gradients[layer][node] * _neurons[layer - 1][weight] * LearningRate;
                }
            }

            return gradients;
        }
    }
    */
}