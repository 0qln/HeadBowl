using HeadBowl.Helpers;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace HeadBowl.Layers
{
    public class SigmoidLayerBuilder<TPrecision> : ILayerBuilder<TPrecision>
    {
        public int Size => _instance.Size;

        int ILayerBuilder<TPrecision>.Size => throw new NotImplementedException();

        private readonly ILayer<TPrecision> _instance;
        private ILayer<TPrecision>? _next, _prev;



        public SigmoidLayerBuilder(int size)
        {
            _instance = new SigmoidLayer<TPrecision>(size);
        }


        void ILayerBuilder<TPrecision>.SetNext(ILayer<TPrecision>? layer)
        {
            _next = layer;
        }

        void ILayerBuilder<TPrecision>.SetPrev(ILayer<TPrecision>? layer)
        {
            _prev = layer;
        }

        ILayer<TPrecision> ILayerBuilder<TPrecision>.Build()
        {
            _instance._InitInNet(_prev, _next);
            return _instance;
        }

        ILayer<TPrecision> ILayerBuilder<TPrecision>.Instance()
        {
            return _instance;
        }
    }


    public class SigmoidLayer<TPrecision> : ILayer<TPrecision>
    {
        private ILayer<TPrecision> _layer;



        public int Size => _layer.Size;
        public TPrecision[,] Weights => _layer.Weights;
        public TPrecision[] Activations => _layer.Activations;
        public TPrecision[] Gradients => _layer.Gradients;
        public bool IsOutputLayer => _layer.IsOutputLayer;
        public bool IsInputLayer => _layer.IsInputLayer;


        public void ApplyGradients()
            => _layer.ApplyGradients();

        public void Forward(in TPrecision[]? inputs)
            => _layer.Forward(inputs);

        public void GenerateGradients(in TPrecision[]? expectedNNOutputs = null)
            => _layer.GenerateGradients(expectedNNOutputs);

        void ILayer<TPrecision>._InitInNet(ILayer<TPrecision>? prev, ILayer<TPrecision>? next)
            => _layer._InitInNet(prev, next);


        public SigmoidLayer(int size)
        {
            _layer =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_16bit((ILayer<Half>)prevLayer, (ILayer<Half>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_32bit((ILayer<Single>)prevLayer, (ILayer<Single>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new SigmoidLayer_64bit(size) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_128bit((ILayer<Decimal>)prevLayer, (ILayer<Decimal>)nextLayer, size)*/
                : throw new NotImplementedException();
        }

        public SigmoidLayer(
            ILayer<TPrecision> prevLayer,
            ILayer<TPrecision> nextLayer,
            int size)
        {
            _layer =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_16bit((ILayer<Half>)prevLayer, (ILayer<Half>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_32bit((ILayer<Single>)prevLayer, (ILayer<Single>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new SigmoidLayer_64bit(size, (ILayer<double>)prevLayer, (ILayer<double>)nextLayer) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_128bit((ILayer<Decimal>)prevLayer, (ILayer<Decimal>)nextLayer, size)*/
                : throw new NotImplementedException();
        }
    }


    internal class SigmoidLayer_64bit : LayerBase_64bit, ILayer<double>
    {
        public SigmoidLayer_64bit(int size, ILayer<double>? prevLayer, ILayer<double>? nextLayer)
            : base(size, prevLayer, nextLayer)
        {
        }

        public SigmoidLayer_64bit(int size)
            : base(size)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override double Activation(double input) => 1 / (1 + Math.Exp(-input));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override double ActivationDerivative(double input) => input * (1 - input); // when this is used, the input is already sigmoided.
    }

}
