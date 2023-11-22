using HeadBowl.Optimizers;

namespace HeadBowl.Layers
{
    public interface ILayerBuilder<T>
    {
        public int Size { get; }
        internal void SetNext(ILayer<T>? layer);
        internal void SetPrev(ILayer<T>? layer);
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

        void ILayerBuilder<TPrecision>.SetNext(ILayer<TPrecision>? layer) => _next = layer;
        void ILayerBuilder<TPrecision>.SetPrev(ILayer<TPrecision>? layer) => _prev = layer;

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

    public class FullyConnectedLayer<TPrecision, TActivation> : LayerBuilderBase<TPrecision>
        where TActivation : IActivationType, new()
    {
        public FullyConnectedLayer(int size, IOptimizer<TPrecision>? optimizer = null) 
            : base(
                typeof(TPrecision) == typeof(double) ? (ILayer<TPrecision>)new FullyConnectedLayer_64bit
                  (size, new TActivation().GetInstance<double>(), (IOptimizer<double>)(optimizer ?? new None<TPrecision>())) :
                //typeof(TPrecision) == typeof(float) ? (ILayer<TPrecision>)new FullyConnectedSigmoidLayer_32bit(size, new TActivation().GetInstance<float>()) :
                throw new NotImplementedException()
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
