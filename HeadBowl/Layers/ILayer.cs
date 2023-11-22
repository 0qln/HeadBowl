using HeadBowl.Helpers;
using HeadBowl.Optimizers;
using System.Diagnostics;
using System.IO.Pipes;

namespace HeadBowl.Layers
{
    public interface ILayer { }

    public interface ILayer<T> : ILayer
    {
        IActivation<T> Activation { get; }
        IOptimizer<T> Optimizer { get; }

        bool IsOutputLayer { get; }
        bool IsInputLayer { get; }

        bool EnableParallelProcessing { get; set; }
        bool ExperimentalFeature { get; set; }

        int Size { get; }

        internal void _InitInNet(ILayer<T>? prev, ILayer<T>? next);

        // 1.
        void Forward();
        // 2.
        void GenerateGradients();
        // 3.
        void ApplyOptimizer();
        // 4.
        void ApplyGradients();


        Array Inputs { set; }
        Array GradientDependencies { set; }

        Array? Activations { get; set; }
        Array? Gradients { get; }
        Array Weights { get; }

        Array LearningRates { get; }
    }
}
