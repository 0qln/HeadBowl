using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.OLD.Layers
{
    public interface ILayerBuilder
    {
        internal ILayer _build(ILayer prevLayer);
    }

    public interface ILayerBuilder<TFloat>
    {
        internal ILayer<TFloat> _build(ILayer<TFloat> prevLayer);
    }
}
