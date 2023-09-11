using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
{
    public interface IComputationalLayer<TFloat> : ILayer<TFloat>
    {
        /// <summary>
        /// Feed forward
        /// </summary>
        /// <returns></returns>
        public TFloat[] Forward();


        /// <summary>
        /// The backpropagation funtion for the hidden layers.
        /// </summary>
        /// <param name="expected"></param>
        /// <returns>The gradients for the previous layer</returns>
        public TFloat[] Backward();


        /// <summary>
        /// The backpropagation function for the output layer.
        /// </summary>
        /// <param name="expected"></param>
        /// <returns>The expected outputs of the network</returns>
        public TFloat[] Backward(TFloat[] expected);


        /// <summary>
        /// The computed gradients from the last backward pass.
        /// </summary>
        public TFloat[] LastGradients { get; }


        /// <summary>
        /// [previous layer, this layer]
        /// </summary>
        public TFloat[,] Weights { get; }
    }
}
