using HeadBowl.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{

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
        }
    }
}
