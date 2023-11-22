using HeadBowl.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public class None<TPrecision> : IOptimizer<TPrecision>
    {
        public void Optimize(ILayer<TPrecision> data) { }
        public IOptimizer<TPrecision> Clone() => new None<TPrecision>();
    }
}
