using HeadBowl.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public interface IOptimizerType
    {
        public IOptimizer<TPrecision> GetInstance<TPrecision>();
    }


    public interface IOptimizer<T>
    {
        void Optimize(ILayer<T> data);
    }
}
