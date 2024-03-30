using HeadBowl.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public static partial class Optimizers
    {
        public static Func<IAdam<TPrecision>> Adam<TPrecision>(
            TPrecision alpha,
            TPrecision beta1,
            TPrecision beta2,
            TPrecision epsilon)
        {
            return
                typeof(TPrecision) == typeof(double) ? () => (IAdam<TPrecision>)new Adam_64bit((dynamic)alpha!, (dynamic)beta1!, (dynamic)beta2!, (dynamic)epsilon!)
                : throw new NotImplementedException();
        }
    }

    public interface IAdam<T> : IOptimizer<T>
    {
    }

    internal class Adam_64bit : IAdam<double>
    // store all values adam needs for it's optimizations
    // e.g. cache the gradients of previous epochs...
    {
        private readonly double _alpha;
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _epsilon;

        double[]? IOptimizer<double>.BiasUpdates => throw new NotImplementedException();

        double[,]? IOptimizer<double>.WeightUpdates => throw new NotImplementedException();

        public Adam_64bit(double alpha, double beta1, double beta2, double epsilon)
        // receive meta parameters
        {
            _alpha = alpha;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
        }

        public void Optimize(ILayer<double> data)
        // modify ILayer gradients in adam fashion
        {

            for (int g = 0; g <= data.Gradients.Length; g++)
            {
                //var m = _beta1 * m + (1 - _beta1) * g;
            }

        }

        public IOptimizer<double> Clone()
        {
            return new Adam_64bit(_alpha, _beta1, _beta2, _epsilon);
        }

        public void Load(ILayer<double> data)
        {
            throw new NotImplementedException();
        }
    }
}
