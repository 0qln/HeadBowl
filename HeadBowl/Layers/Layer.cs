﻿using HeadBowl.Activations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
{
    public class Layer<TActivation, TFloat> : IComputationalLayer<TFloat>
        where TFloat : struct
        where TActivation : IActivationFunction<TFloat>, new()
    {
        public IComputationalLayer<TFloat> NextLayer { get; private set; }
        public ILayer<TFloat> PrevLayer { get; private set; }
        public int Size { get; private set; }
        public TFloat[] Biases { get; private set; }
        public TFloat[,] Weights { get; private set; }
        public TFloat[] Values { get; private set; }

        // Learning
        public TFloat[] LastGradients { get; private set; }
        public TFloat LearningRate { get; set; }
        public TFloat Cost { get; private set; }

        private readonly TActivation _activate = new();


        public Layer (
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
        internal Layer() { }
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


        /// <summary><inheritdoc/></summary>
        public TFloat[] Backward(TFloat[] expected)
        {
            var gradients = new TFloat[Size];

            // calculate gradient
            for (int node = 0; node < Size; node++)
            {
                gradients[node] = ((dynamic)Values[node] - expected[node]) * _activate.Backward(Values[node]);
            }

            // apply gradients for the last layer
            for (int node = 0; node < Size; node++)
            {
                //calculates the w' and b' for the last layer in the network
                Biases[node] -= (dynamic)gradients[node] * LearningRate;

                for (int weight = 0; weight < PrevLayer.Size; weight++)
                {
                    Weights[node, weight] -= (dynamic)gradients[node] * PrevLayer.Values[weight] * LearningRate;
                }
            }

            return gradients;
        }

        /// <summary><inheritdoc/></summary>
        public TFloat[] Backward()
        {
            var gradients = new TFloat[Size];

            // calculate gradients
            for (int node = 0; node < Size; node++)
            {
                for (int nextLayerNode = 0; nextLayerNode < NextLayer.Size; nextLayerNode++)
                {
                    gradients[node] += (dynamic)NextLayer.LastGradients[nextLayerNode] * NextLayer.Weights[nextLayerNode, node];
                }
                gradients[node] *= (dynamic)_activate.Backward(Values[node]);
            }

            //iterate over outputs of layer
            for (int node = 0; node < Size; node++)
            {
                // modify biases of network
                Biases[node] -= (dynamic)gradients[node] * LearningRate;
                // iterate over inputs to layer
                for (int weight = 0; weight < PrevLayer.Size; weight++)
                {
                    // modify weights of network
                    Weights[node, weight] -= (dynamic)NextLayer.LastGradients[node] * Values[weight] * LearningRate;
                }
            }

            return gradients;
        }
    }
}
