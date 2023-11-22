using HeadBowl.Helpers;
using System.Diagnostics;
using System.IO.Pipes;

namespace HeadBowl.Layers
{
    public interface ILayer<T>
    {
        bool IsOutputLayer { get; }
        bool IsInputLayer { get; }

        bool EnableParallelProcessing { get; set; }

        int Size { get; }

        internal void _InitInNet(ILayer<T>? prev, ILayer<T>? next);
        void Forward();
        void GenerateGradients();
        void ApplyGradients();

        Array Inputs { set; }
        Array GradientDependencies { set; }

        Array? Activations { get; set; }
        Array? Gradients { get; }
        Array Weights { get; }
    }
}
