using HeadBowl.Layers;
using Microsoft.Toolkit.HighPerformance;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public static partial class Optimizers
    {
        public static IOptimizer<TPrecision> None<TPrecision>()
        {
            return new None<TPrecision>();
        }
    }

    public class None<TPrecision> : IOptimizer<TPrecision>
    {
        TPrecision[]? _biasUpdates;
        TPrecision[,]? _weightUpdates;

        TPrecision[]? IOptimizer<TPrecision>.BiasUpdates => _biasUpdates;
        TPrecision[,]? IOptimizer<TPrecision>.WeightUpdates => _weightUpdates;

        public void Optimize(ILayer<TPrecision> data)
        {
            if (_biasUpdates is not null)
                Array.Fill(_biasUpdates, default);
            if (_weightUpdates is not null)
                new Span2D<TPrecision>(_weightUpdates).Fill(default!);
        }

        public IOptimizer<TPrecision> Clone() => new None<TPrecision>();
        
        public void Load(ILayer<TPrecision> data)
        {
            _biasUpdates = new TPrecision[data.Biases.Length];
            _weightUpdates = new TPrecision[data.Weights.GetLength(0), data.Weights.GetLength(1)];
        }
        
    }
}
