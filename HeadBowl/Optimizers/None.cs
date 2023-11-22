using HeadBowl.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public class None : IOptimizer<object?>
    {
        public void Optimize(ILayer<object?> data)
        {
            throw new NotImplementedException();
        }
    }
}
