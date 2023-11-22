using HeadBowl.Helpers;
using HeadBowl.Optimizers;
using System.Diagnostics;
using System.IO.Pipes;

namespace HeadBowl.Layers
{
    public interface ILayer<T>
    {
        IOptimizer<T> Optimizer { get; }

        bool IsOutputLayer { get; }
        bool IsInputLayer { get; }

        bool EnableParallelProcessing { get; set; }
        bool ExperimentalFeature { get; set; }

        int Size { get; }

        internal void _InitInNet(ILayer<T>? prev, ILayer<T>? next);
        
        abstract public T Activation(T input);
        abstract public T ActivationDerivative(T input);

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
    }
}
