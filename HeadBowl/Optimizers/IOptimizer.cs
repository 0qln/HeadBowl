using HeadBowl.Layers;
using HeadBowl.Optimizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public static partial class Optimizers { }

    public interface IOptimizer<TPrecision> 
    {
        public IOptimizer<TPrecision> Clone();
        public void Optimize(ILayer<TPrecision> data);
        public void Load(ILayer<TPrecision> data);
        public TPrecision[]? BiasUpdates { get; }
        public TPrecision[,]? WeightUpdates { get; }
    }

}