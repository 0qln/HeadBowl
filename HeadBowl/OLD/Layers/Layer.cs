using System.Numerics;
using HeadBowl.OLD.Activations;

namespace HeadBowl.OLD.Layers
{
    public class Layer<T, TFloat> : ILayer<TFloat>
        where TFloat : struct
        where T : IActivationFunction<TFloat>, new()
    {
        public ILayer<TFloat> PrevLayer { get; init; }
        public int Size { get; init; }
        public TFloat[] Biases { get; }
        public TFloat[,] Weights { get; }
        public TFloat[] Values { get; }
        public TFloat LearningRate { get; set; }
        public TFloat Cost { get; private set; }

        private readonly T _activate = new();


        public Layer(
            ILayer<TFloat> prevLayer,
            int size)
        {
            PrevLayer = prevLayer;
            Size = size;
            Biases = new TFloat[Size];
            Weights = new TFloat[PrevLayer.Size, Size];
            Values = new TFloat[Size];
        }


        public TFloat[] Forward()
        {
            var result = new TFloat[Size];

            for (int i = 0; i < Size; i++)
            {
                result[i] = Biases[i];

                for (int j = 0; j < PrevLayer.Size; j++)
                {
                    result[i] += (dynamic)Weights[j, i] * (dynamic)PrevLayer.Values[j];
                }

                result[i] = _activate.Forward(result[i]);
            }

            return result;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="expected"></param>
        /// <returns>The gradients for the previous layer</returns>
        public TFloat[] Backward(TFloat[] expected)
        {
            // Calculate the cost 
            //Cost = Loss.Loss.MSE(Values, expected);

            var gradients = new TFloat[Size];
            // calculate gradient
            for (int i = 0; i < Size; i++)
            {
                gradients[i] = ((dynamic)Values[i] - expected[i]) * _activate.Backward(Values[i]);
            }

            // apply gradients for the last layer
            for (int i = 0; i < Size; i++)
            {
                //calculates the w' and b' for the last layer in the network
                Biases[i] -= (dynamic)gradients[i] * LearningRate;

                for (int j = 0; j < PrevLayer.Size; j++)
                {
                    Weights[j, i] -= (dynamic)gradients[i] * PrevLayer.Values[j] * LearningRate;
                }
            }

            return gradients;
        }
    }
}