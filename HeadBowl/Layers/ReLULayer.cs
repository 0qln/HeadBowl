using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Drawing;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using HeadBowl.Helpers;

namespace HeadBowl.Layers
{
    public class ReLULayerBuilder<TPrecision> : ILayerBuilder<TPrecision>
    {
        public int Size => _instance.Size;

        int ILayerBuilder<TPrecision>.Size => throw new NotImplementedException();

        private readonly ILayer<TPrecision> _instance;
        private ILayer<TPrecision>? _next, _prev;



        public ReLULayerBuilder(int size)
        {
            _instance = new ReLULayer<TPrecision>(size);
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


    public class ReLULayer<TPrecision> : ILayer<TPrecision>
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


        public ReLULayer(int size)
        {
            _layer =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_16bit((ILayer<Half>)prevLayer, (ILayer<Half>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_32bit((ILayer<Single>)prevLayer, (ILayer<Single>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new ReLULayer_64bit(size) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_128bit((ILayer<Decimal>)prevLayer, (ILayer<Decimal>)nextLayer, size)*/
                : throw new NotImplementedException();
        }

        public ReLULayer(
            ILayer<TPrecision> prevLayer,
            ILayer<TPrecision> nextLayer,
            int size)
        {
            _layer =
                typeof(TPrecision) == typeof(Half) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_16bit((ILayer<Half>)prevLayer, (ILayer<Half>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(float) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_32bit((ILayer<Single>)prevLayer, (ILayer<Single>)nextLayer, size)*/ :
                typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new ReLULayer_64bit(size, (ILayer<double>)prevLayer, (ILayer<double>)nextLayer) :
                typeof(TPrecision) == typeof(decimal) ? throw new NotImplementedException() /*(ILayer<T>)new ReLULayer_128bit((ILayer<Decimal>)prevLayer, (ILayer<Decimal>)nextLayer, size)*/
                : throw new NotImplementedException();
        }
    }


    internal class ReLULayer_64bit : LayerBase_64bit, ILayer<double>
    {
        public ReLULayer_64bit(int size, ILayer<double>? prevLayer, ILayer<double>? nextLayer)
            : base (size, prevLayer, nextLayer)
        {
        }

        public ReLULayer_64bit(int size)
            : base (size)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override double Activation(double input) => Math.Max(input, 0);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override double ActivationDerivative(double input) => input < 0 ? 0 : 1;

    }

}
