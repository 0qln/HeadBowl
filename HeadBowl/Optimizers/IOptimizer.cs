using HeadBowl.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public interface IOptimizer<TPrecision> 
    {
        public IOptimizer<TPrecision> Clone();
        public void Optimize(ILayer<TPrecision> data);
    }

    public static class Optimizers
    {
        public static IOptimizer<TPrecision> None<TPrecision>()
        {
            return new None<TPrecision>();
        }

        public static IAdam<TPrecision> Adam<TPrecision>(
            TPrecision alpha,
            TPrecision beta1,
            TPrecision beta2,
            TPrecision epsilon)
        {
            return
                typeof(TPrecision) == typeof(double) ? (IAdam<TPrecision>)new Adam_64bit((dynamic)alpha!, (dynamic)beta1!, (dynamic)beta2!, (dynamic)epsilon!)
                : throw new NotImplementedException();
        }
    }
}
