using HeadBowl.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Optimizers
{
    public static partial class Optimizers
    {
        public static IMomentum<T> Momentum<T>(T beta)
        {
            return
                typeof(T) == typeof(double) ? (dynamic) new Momentum_64bit((dynamic)beta!) :
                throw new NotImplementedException();
        }
    }

    public interface IMomentum<T> : IOptimizer<T>
    {
    }

    internal class Momentum_64bit : IMomentum<double>
    {
        private readonly double _beta;

        double[]? _biasUpdates;
        double[,]? _weightUpdates;

        double[]? IOptimizer<double>.BiasUpdates => _biasUpdates;
        double[,]? IOptimizer<double>.WeightUpdates => _weightUpdates;


        /// <summary></summary>
        /// <param name="beta">A number between 0 and 1.</param>
        public Momentum_64bit(double beta)
        {
            _beta = beta;
        }

        public void Load(ILayer<double> data)
        {
            _biasUpdates = new double[data.Biases.Length];
            _weightUpdates = new double[data.Weights.GetLength(0), data.Weights.GetLength(1)];
        }

        public void Optimize(ILayer<double> data)
        {
            for (int i = 0; i < _biasUpdates?.Length; i++)
                _biasUpdates[i] *= _beta;

            for (int i = 0; i < _weightUpdates?.GetLength(0); i++)
                for (int j = 0; j < _weightUpdates?.GetLength(1); j++)
                    _weightUpdates[i, j] *= _beta;
        }

        public IOptimizer<double> Clone()
        {
            return new Momentum_64bit(_beta);
        }
    }
}
