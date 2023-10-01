using HeadBowl.Helpers;
using System.Diagnostics;
using System.IO.Pipes;

namespace HeadBowl.Layers
{
    public interface ILayer<T>
    {
        // Properties
        bool IsOutputLayer { get; }
        bool IsInputLayer { get; }

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
