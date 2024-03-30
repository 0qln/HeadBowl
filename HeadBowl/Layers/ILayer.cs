using HeadBowl.Helpers;
using HeadBowl.Optimizers;
using System.Diagnostics;
using System.IO.Pipes;

namespace HeadBowl.Layers
{
    public interface ILayer 
    {
        // Training Pipeline:
        // ------------------

        /// <summary>
        /// 1. Forward the training data.
        /// </summary>
        void Forward();

        /// <summary>
        /// 2. Calculate the gradients of the network.
        /// </summary>
        void GenerateGradients();

        /// <summary>
        /// 3. Apply optimizer to the parameter updates.
        /// </summary>
        void ApplyOptimizer();

        /// <summary>
        /// 4. Apply default gradient descent to the parameter updates.
        /// </summary>
        void ApplyGradients();

        /// <summary>
        /// 5. Update the layer parameters.
        /// </summary>
        void UpdateParamaters();
    }

    public interface ILayer<TPrecision> : ILayer
    {
        // Layer type
        bool IsOutputLayer { get; }
        bool IsInputLayer { get; }

        // Misc. Features
        bool EnableParallelProcessing { get; set; }
        bool ExperimentalFeature { get; set; }

        ///Helper functions
        internal void InitInNet(ILayer<TPrecision>? prev, ILayer<TPrecision>? next);
        internal ILayerBuilder<TPrecision> ToRawBuilder();

        // Details
        IActivation<TPrecision> Activation { get; }
        IOptimizer<TPrecision> Optimizer { get; set; }
        int Size { get; }
        Array Inputs { set; }
        Array GradientDependencies { set; }
        Array? Activations { get; set; }
        Array? Gradients { get; }
        Array Weights { get; set; }
        Array Biases { get; set; }
        Array LearningRates { get; }
    }
}
