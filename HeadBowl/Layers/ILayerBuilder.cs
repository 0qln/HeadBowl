using HeadBowl.Optimizers;

namespace HeadBowl.Layers
{
    public interface ILayerBuilder<T>
    {
        public int Size { get; }
        internal ILayerBuilder<T> SetNext(ILayer<T>? layer);
        internal ILayerBuilder<T> SetPrev(ILayer<T>? layer);
        public ILayer<T> Build();
        public ILayer<T> Instance();
    }

    public abstract class LayerBuilderBase<TPrecision> : ILayerBuilder<TPrecision>
    {
        public int Size => _instance.Size;

        protected ILayer<TPrecision> _instance;

        protected ILayer<TPrecision>? _next, _prev;
        
        protected LayerBuilderBase(ILayer<TPrecision> instance)
        {
            _instance = instance;
        }

        ILayerBuilder<TPrecision> ILayerBuilder<TPrecision>.SetNext(ILayer<TPrecision>? layer)
        {
            _next = layer;
            return this;
        }

        ILayerBuilder<TPrecision> ILayerBuilder<TPrecision>.SetPrev(ILayer<TPrecision>? layer)
        {
            _prev = layer;
            return this;
        }

        ILayer<TPrecision> ILayerBuilder<TPrecision>.Build()
        {
            _instance.InitInNet(_prev, _next);
            return _instance;
        }

        ILayer<TPrecision> ILayerBuilder<TPrecision>.Instance()
        {
            return _instance;
        }
    }

    public class FullyConnectedLayer<TPrecision, TActivation> : LayerBuilderBase<TPrecision>
        where TActivation : IActivationType, new()
    {
        public FullyConnectedLayer(int size, Func<IOptimizer<TPrecision>>? optimizer = null) 
            : base(
                typeof(TPrecision) == typeof(double) 
                    ? (ILayer<TPrecision>) new FullyConnectedLayer_64bit(
                        size: size, 
                        activation: new TActivation().GetInstance<double>(), 
                        optimizer: (IOptimizer<double>)(optimizer?.Invoke() ?? new None<TPrecision>())) 
                    : throw new NotImplementedException()
            )
        { 
        }
    }

    //public class ConvolutionLayer<TPrecision> : LayerBuilderBase<TPrecision>
    //{
    //    public ConvolutionLayer(ActivationType activation, int inputDepth, int outputDepth, int filterExtend, int stride, int zeroPadding)
    //        : base(activation switch
    //        {
    //            ActivationType.ReLU =>
    //                typeof(TPrecision) == typeof(double)? (ILayer<TPrecision>) new ConvReLULayer_64bit(inputDepth, outputDepth, filterExtend, stride, zeroPadding)
    //                : throw new NotImplementedException(),

    //            _ => throw new NotImplementedException()
    //        })
    //    { 
    //    }s
    //}
}
