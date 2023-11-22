using HeadBowl.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public class Adam : IOptimizerType
    {
        public IOptimizer<TPrecision> GetInstance<TPrecision>()
        {
            return 
                typeof(TPrecision) == typeof(double) ? (IOptimizer<TPrecision>)new Adam_64bit() :
                typeof(TPrecision) == typeof(float) ? (IOptimizer<TPrecision>)new Adam_32bit() :
                throw new NotImplementedException();
        }
    }
    public interface IAdam<T> : IOptimizer<T>
    {
    }
    internal class Adam_64bit : IAdam<double>
    // store all values adam needs for it's optimizations
    // e.g. cache the gradients of previous epochs...
    {
        public void Optimize(ILayer<double> data)
        // modify ILayer gradients in adam fashion
        {
        }
    }
    internal class Adam_32bit : IAdam<double>
    {
        public void Optimize(ILayer<double> data)
        {
        }
    }
}
